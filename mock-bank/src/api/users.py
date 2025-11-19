import base64
import io
from decimal import Decimal
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import select, func

from db.models import User, Account, Beneficiary
from logging_config import get_logger
from .deps import get_db
from .schemas import (
    AudioEmbeddingOut,
    AudioEmbeddingUpdate,
    LoginRequest,
    LoginResult,
    UserCreate,
    UserOut,
    AccountOut,
    BeneficiaryOut,
    BeneficiaryCreate,
)
from .serializers import serialize_account, serialize_user, serialize_beneficiary

logger = get_logger("mock_bank.api.users")

router = APIRouter(tags=["users"])


# Optional voice embedding support via Resemblyzer
try:  # pragma: no cover - optional dependency
    from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
    import soundfile as sf  # type: ignore

    VOICE_EMBEDDER_AVAILABLE = True
    _voice_encoder = VoiceEncoder()
    logger.info("Voice embedding: Resemblyzer encoder initialized in users API")
except Exception as e:  # pragma: no cover - optional dependency
    VOICE_EMBEDDER_AVAILABLE = False
    _voice_encoder = None
    logger.warning("Voice embedding libraries not available: %s", e)


def _extract_voice_embedding(audio_bytes: bytes) -> Optional[list[float]]:
    """
    Create a voice embedding vector from raw audio bytes.
    Uses Resemblyzer when available, otherwise falls back to a simple stub.
    """
    if not audio_bytes:
        return None

    if VOICE_EMBEDDER_AVAILABLE and _voice_encoder is not None:
        try:
            # Attempt to read audio bytes (expects a standard audio container, e.g. WAV).
            wav, sr = sf.read(io.BytesIO(audio_bytes))
            wav = preprocess_wav(wav, source_sr=sr)
            emb = _voice_encoder.embed_utterance(wav)
            return [float(x) for x in emb]
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Voice embedding extraction failed; falling back to stub: %s", e)

    # Stub fallback: deterministic vector based on length only.
    length = float(len(audio_bytes))
    return [length, 0.0, 0.0]


@router.get("/list_users", response_model=List[UserOut])
async def list_users(limit: int = 50, offset: int = 0, db=Depends(get_db)):
    """
    Return a paginated list of users.
    """
    logger.info("Listing users limit=%s offset=%s", limit, offset)
    stmt = select(User).order_by(User.created_at.desc()).limit(limit).offset(offset)
    res = await db.execute(stmt)
    users = res.scalars().all()
    return [serialize_user(u) for u in users]


@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: UUID, db=Depends(get_db)):
    """
    Fetch a single user by id.
    """
    logger.info("Fetching user user_id=%s", user_id)
    stmt = select(User).where(User.user_id == user_id)
    res = await db.execute(stmt)
    u = res.scalars().first()
    if not u:
        logger.warning("User not found user_id=%s", user_id)
        raise HTTPException(status_code=404, detail="User not found")
    return serialize_user(u)


@router.get("/users/{user_id}/accounts", response_model=List[AccountOut])
async def get_user_accounts(user_id: UUID, db=Depends(get_db)):
    """
    Return all accounts linked to a user.
    """
    logger.info("Fetching accounts for user_id=%s", user_id)
    stmt = select(Account).where(Account.user_id == user_id)
    res = await db.execute(stmt)
    accounts = res.scalars().all()
    return [serialize_account(a) for a in accounts]


@router.post("/users", response_model=UserOut)
async def create_user(payload: UserCreate, db=Depends(get_db)):
    """
    Create a new user with username + passphrase (+ optional email and audio_embedding).
    """
    # Normalise username/passphrase to lower-case for robustness with STT
    username_norm = (payload.username or "").strip().lower()
    passphrase_norm = (payload.passphrase or "").strip().lower()

    logger.info("Creating user username=%s", username_norm)

    # Check if username already exists (case-insensitive)
    stmt = select(User).where(func.lower(User.username) == username_norm)
    res = await db.execute(stmt)
    exists = res.scalars().first()
    if exists:
        logger.warning("Username already exists username=%s", username_norm)
        raise HTTPException(status_code=409, detail="Username already exists")

    # Basic inferred email if not provided
    email = payload.email or f"{username_norm}@example.com"

    # For this prototype, we treat passphrase as the source for both passphrase and password_hash.
    password_hash = passphrase_norm

    # Determine audio_embedding:
    #  - If provided explicitly, trust it.
    #  - Else, if audio_data present, decode and compute embedding.
    audio_embedding = payload.audio_embedding
    if audio_embedding is None and payload.audio_data:
        try:
            audio_bytes = base64.b64decode(payload.audio_data)
            audio_embedding = _extract_voice_embedding(audio_bytes)
            logger.info(
                "create_user: computed audio embedding (len=%d) for username=%s",
                len(audio_embedding or []),
                username_norm,
            )
        except Exception as e:
            logger.exception("create_user: failed to decode audio_data: %s", e)

    # Generate a deterministic default savings account number for this prototype.
    # We use the current count of accounts to form ACC000001, ACC000002, ...
    acct_number = None
    try:
        count_stmt = select(func.count()).select_from(Account)
        count_res = await db.execute(count_stmt)
        current_count = count_res.scalar_one() or 0
        acct_number = f"ACC{current_count + 1:06d}"
    except Exception as e:
        logger.exception("create_user: failed to compute next account number; falling back to random: %s", e)
        # very low collision risk; acceptable for prototype
        acct_number = f"ACC{uuid4().hex[:8].upper()}"

    # Create user first with an explicit UUID so we can safely link the default
    # account without relying on server-side key generation.
    new_user_id = uuid4()

    u = User(
        user_id=new_user_id,
        username=username_norm,
        email=email,
        password_hash=password_hash,
        full_name=payload.full_name,
        phone_number=payload.phone_number,
        address=payload.address,
        # date_of_birth is accepted as an ISO string; parsing can be added if needed.
        passphrase=passphrase_norm,
        audio_embedding=audio_embedding,
    )

    default_balance = Decimal("5000.00")
    # Add user and flush so we have a concrete user_id for the FK
    db.add(u)
    await db.flush()

    acct = Account(
        account_id=uuid4(),
        account_number=acct_number,
        user_id=new_user_id,
        account_type="savings",
        currency="USD",
        balance=default_balance,
        available_balance=default_balance,
        status="active",
    )
    db.add(acct)

    # Commit both user and default account in a single transaction
    await db.commit()
    await db.refresh(u)
    await db.refresh(acct)
    logger.info(
        "Created user user_id=%s username=%s with default savings account %s balance=%s",
        u.user_id,
        u.username,
        acct.account_number,
        acct.balance,
    )
    return serialize_user(u)


@router.post("/register", response_model=UserOut)
async def register_user(payload: UserCreate, db=Depends(get_db)):
    """
    Convenience alias for POST /api/users (user registration).
    """
    return await create_user(payload, db)


@router.post("/login", response_model=LoginResult)
async def login(payload: LoginRequest, db=Depends(get_db)):
    """
    Validate username + passphrase and return basic auth info.
    """
    username_norm = (payload.username or "").strip().lower()
    passphrase_norm = (payload.passphrase or "").strip().lower()

    logger.info("Login attempt username=%s", username_norm)
    stmt = select(User).where(func.lower(User.username) == username_norm)
    res = await db.execute(stmt)
    user = res.scalars().first()
    if not user:
        logger.warning("Login failed - user not found username=%s", username_norm)
        raise HTTPException(status_code=404, detail="User not found")

    if not user.passphrase or user.passphrase.lower() != passphrase_norm:
        logger.warning("Login failed - invalid passphrase username=%s", username_norm)
        raise HTTPException(status_code=401, detail="Invalid passphrase")

    logger.info("Login successful user_id=%s username=%s", user.user_id, user.username)
    return LoginResult(
        user_id=user.user_id,
        username=user.username,
        status="ok",
        message="Login successful",
        has_audio_embedding=bool(getattr(user, "audio_embedding", None)),
    )


@router.put("/users/{user_id}/audio-embedding", response_model=UserOut)
async def update_audio_embedding(user_id: UUID, payload: AudioEmbeddingUpdate, db=Depends(get_db)):
    """
    Store or replace the audio_embedding for a user.
    """
    logger.info("Updating audio_embedding for user_id=%s", user_id)
    stmt = select(User).where(User.user_id == user_id)
    res = await db.execute(stmt)
    user = res.scalars().first()
    if not user:
        logger.warning("User not found for audio-embedding update user_id=%s", user_id)
        raise HTTPException(status_code=404, detail="User not found")

    user.audio_embedding = payload.audio_embedding
    await db.commit()
    await db.refresh(user)
    return serialize_user(user)


@router.get("/users/{user_id}/audio-embedding", response_model=AudioEmbeddingOut)
async def get_audio_embedding(user_id: UUID, db=Depends(get_db)):
    """
    Retrieve the stored audio_embedding for a user.
    """
    logger.info("Fetching audio_embedding for user_id=%s", user_id)
    stmt = select(User).where(User.user_id == user_id)
    res = await db.execute(stmt)
    user = res.scalars().first()
    if not user:
        logger.warning("User not found for audio-embedding fetch user_id=%s", user_id)
        raise HTTPException(status_code=404, detail="User not found")

    if user.audio_embedding is None:
        logger.warning("No audio embedding stored for user_id=%s", user_id)
        raise HTTPException(status_code=404, detail="No audio embedding stored for this user")

    return AudioEmbeddingOut(user_id=user.user_id, audio_embedding=user.audio_embedding)


@router.get("/users/{user_id}/beneficiaries", response_model=List[BeneficiaryOut])
async def get_user_beneficiaries(
    user_id: UUID,
    limit: int = 50,
    offset: int = 0,
    db=Depends(get_db),
):
    """
    Return a paginated list of beneficiaries (saved payees) for a user.
    """
    logger.info("Fetching beneficiaries for user_id=%s limit=%s offset=%s", user_id, limit, offset)
    # Ensure user exists
    stmt_user = select(User).where(User.user_id == user_id)
    res_user = await db.execute(stmt_user)
    u = res_user.scalars().first()
    if not u:
        logger.warning("User not found when fetching beneficiaries user_id=%s", user_id)
        raise HTTPException(status_code=404, detail="User not found")

    stmt = (
        select(Beneficiary)
        .where(Beneficiary.user_id == user_id)
        .order_by(Beneficiary.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    res = await db.execute(stmt)
    beneficiaries = res.scalars().all()
    return [serialize_beneficiary(b) for b in beneficiaries]


@router.post("/users/{user_id}/beneficiaries", response_model=BeneficiaryOut)
async def create_beneficiary_for_user(
    user_id: UUID,
    payload: BeneficiaryCreate,
    db=Depends(get_db),
):
    """
    Create a new beneficiary (saved payee) for the given user.
    """
    logger.info(
        "Creating beneficiary for user_id=%s nickname=%s account_number=%s",
        user_id,
        payload.nickname,
        payload.account_number,
    )
    # Ensure user exists
    stmt_user = select(User).where(User.user_id == user_id)
    res_user = await db.execute(stmt_user)
    u = res_user.scalars().first()
    if not u:
        logger.warning("User not found when creating beneficiary user_id=%s", user_id)
        raise HTTPException(status_code=404, detail="User not found")

    # Optional duplicate check: avoid duplicates per (user_id, account_number)
    stmt_dup = select(Beneficiary).where(
        Beneficiary.user_id == user_id,
        Beneficiary.beneficiary_account_number == payload.account_number,
    )
    res_dup = await db.execute(stmt_dup)
    existing = res_dup.scalars().first()
    if existing:
        logger.warning(
            "Beneficiary already exists for user_id=%s account_number=%s",
            user_id,
            payload.account_number,
        )
        raise HTTPException(status_code=409, detail="Beneficiary already exists for this account number")

    nickname = payload.nickname or None
    # For now, beneficiary_name mirrors nickname if provided, else falls back to account_number
    beneficiary_name = nickname or payload.account_number

    b = Beneficiary(
        beneficiary_id=uuid4(),
        user_id=user_id,
        beneficiary_account_number=payload.account_number,
        beneficiary_name=beneficiary_name,
        nickname=nickname,
        bank_name=payload.bank_name,
        is_internal=payload.is_internal,
        status="active",
    )
    db.add(b)
    await db.commit()
    await db.refresh(b)

    logger.info(
        "Created beneficiary beneficiary_id=%s user_id=%s nickname=%s account_number=%s",
        b.beneficiary_id,
        user_id,
        b.nickname,
        b.beneficiary_account_number,
    )
    return serialize_beneficiary(b)
