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
    full_name = Column(String(255))
    phone_number = Column(String(20), unique=True)
    preferred_language = Column(String(10))
    status = Column(String(20))
    kyc_status = Column(String(20))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

class Account(Base):
    __tablename__ = "accounts"

    account_id = Column(UUID(as_uuid=True), primary_key=True)
    account_number = Column(String(20), unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    account_type = Column(String(20))
    currency = Column(String(3))
    balance = Column(DECIMAL(15,2))
    available_balance = Column(DECIMAL(15,2))
    status = Column(String(20))
    opened_at = Column(TIMESTAMP)
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
    amount = Column(DECIMAL(15,2))
    currency = Column(String(3))
    fee = Column(DECIMAL(10,2), default=0.00)
    status = Column(String(20))
    description = Column(String)
    metadata_json = Column("metadata",JSONB)
    balance_after = Column(DECIMAL(15,2))
    initiated_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    created_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
