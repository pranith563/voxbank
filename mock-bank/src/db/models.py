# mock-bank/src/db/models.py
from sqlalchemy import Column, String, Date, TIMESTAMP, DECIMAL, JSON, Boolean, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
import sqlalchemy as sa
from db.session import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    # Auth-related fields
    password_hash = Column(String(255), nullable=True)
    # Public profile fields
    full_name = Column(String(255))
    phone_number = Column(String(20), unique=True)
    address = Column(String)
    date_of_birth = Column(Date)
    preferred_language = Column(String(10))
    status = Column(String(20))
    kyc_status = Column(String(20))
    # Plain-text or hashed passphrase for login (demo only).
    # For production, store a salted password hash instead.
    passphrase = Column(String(255), nullable=True)
    # Stored voice embedding for future voice authentication.
    # Represented as JSONB (e.g. list[float]).
    audio_embedding = Column(JSONB, nullable=True)
    last_active = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)


class Account(Base):
    __tablename__ = "accounts"

    account_id = Column(UUID(as_uuid=True), primary_key=True)
    account_number = Column(String(20), unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    account_type = Column(String(20))
    currency = Column(String(3))
    balance = Column(DECIMAL(15, 2))
    available_balance = Column(DECIMAL(15, 2))
    status = Column(String(20))
    interest_rate = Column(DECIMAL(5, 2))
    overdraft_limit = Column(DECIMAL(15, 2))
    opened_at = Column(TIMESTAMP)
    closed_at = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(UUID(as_uuid=True), primary_key=True)
    transaction_reference = Column(String(50), nullable=False)
    account_id = Column(UUID(as_uuid=True), ForeignKey("accounts.account_id"), nullable=False)
    from_account_id = Column(UUID(as_uuid=True), nullable=True)
    to_account_id = Column(UUID(as_uuid=True), nullable=True)
    entry_type = Column(String(10))
    transaction_type = Column(String(30))
    amount = Column(DECIMAL(15, 2))
    currency = Column(String(3))
    fee = Column(DECIMAL(10, 2), default=0.00)
    status = Column(String(20))
    description = Column(String)
    metadata_json = Column("metadata", JSONB)
    balance_after = Column(DECIMAL(15, 2))
    initiated_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)


class Beneficiary(Base):
    __tablename__ = "beneficiaries"

    beneficiary_id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    beneficiary_account_number = Column(String(20), nullable=False)
    beneficiary_name = Column(String(255), nullable=False)
    nickname = Column(String(100), nullable=True)
    bank_name = Column(String(255), nullable=True)
    bank_code = Column(String(20), nullable=True)
    is_internal = Column(Boolean, default=True)
    status = Column(String(20))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)


class Card(Base):
    __tablename__ = "cards"

    card_id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    account_id = Column(UUID(as_uuid=True), ForeignKey("accounts.account_id"), nullable=True)
    card_number = Column(String(32), unique=True, nullable=False)
    card_type = Column(String(20))
    network = Column(String(20))
    last4 = Column(String(4))
    credit_limit = Column(DECIMAL(15, 2))
    current_due = Column(DECIMAL(15, 2))
    min_due = Column(DECIMAL(15, 2))
    due_date = Column(Date)
    status = Column(String(20))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)


class Loan(Base):
    __tablename__ = "loans"

    loan_id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    loan_type = Column(String(50))
    principal_amount = Column(DECIMAL(15, 2))
    outstanding_amount = Column(DECIMAL(15, 2))
    interest_rate = Column(DECIMAL(5, 2))
    emi_amount = Column(DECIMAL(15, 2))
    emi_day_of_month = Column(Integer)
    next_due_date = Column(Date)
    status = Column(String(20))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)


class Reminder(Base):
    __tablename__ = "reminders"

    reminder_id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    reminder_type = Column(String(20))
    title = Column(String(255))
    description = Column(String)
    due_date = Column(TIMESTAMP)
    linked_loan_id = Column(UUID(as_uuid=True), ForeignKey("loans.loan_id"), nullable=True)
    linked_card_id = Column(UUID(as_uuid=True), ForeignKey("cards.card_id"), nullable=True)
    status = Column(String(20))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
