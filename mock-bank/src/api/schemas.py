from decimal import Decimal
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class UserOut(BaseModel):
    user_id: UUID
    username: str
    email: str
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    preferred_language: Optional[str] = None
    status: Optional[str] = None
    kyc_status: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[str] = None
    last_active: Optional[str] = None
    has_audio_embedding: Optional[bool] = None


class AccountOut(BaseModel):
    account_id: UUID
    account_number: str
    user_id: UUID
    account_type: str
    currency: str
    balance: float
    available_balance: Optional[float]
    status: str
    interest_rate: Optional[float] = None
    overdraft_limit: Optional[float] = None
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TransactionOut(BaseModel):
    transaction_id: UUID
    transaction_reference: str
    account_id: UUID
    entry_type: str
    transaction_type: str
    amount: float
    currency: str
    status: str
    description: Optional[str] = None
    fee: Optional[float] = None
    balance_after: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None


class TransferIn(BaseModel):
    from_account_number: str = Field(..., example="ACC-0001")
    to_account_number: str = Field(..., example="ACC-0002")
    amount: Decimal = Field(..., gt=0, example=100.00)
    currency: str = Field("USD")
    initiated_by_user_id: Optional[UUID] = None
    reference: Optional[str] = None


class TransferOut(BaseModel):
    status: str
    transaction_reference: str
    txn_id: Optional[UUID] = None
    message: Optional[str] = None


class UserCreate(BaseModel):
    username: str
    passphrase: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[str] = None  # ISO date string
    audio_embedding: Optional[Any] = None
    audio_data: Optional[str] = None  # base64-encoded audio (optional)


class LoginRequest(BaseModel):
    username: str
    passphrase: str


class LoginResult(BaseModel):
    user_id: UUID
    username: str
    status: str
    message: Optional[str] = None
    has_audio_embedding: bool = False


class AudioEmbeddingUpdate(BaseModel):
    audio_embedding: Any


class AudioEmbeddingOut(BaseModel):
    user_id: UUID
    audio_embedding: Any

