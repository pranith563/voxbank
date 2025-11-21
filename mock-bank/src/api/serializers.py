from typing import Any, Dict

from db.models import User, Account, Transaction, Beneficiary, Card, Loan, Reminder


def serialize_user(u: User) -> Dict[str, Any]:
    return {
        "user_id": str(u.user_id),
        "username": u.username,
        "email": u.email,
        "full_name": u.full_name,
        "phone_number": u.phone_number,
        "preferred_language": u.preferred_language,
        "status": u.status,
        "kyc_status": u.kyc_status,
        "address": getattr(u, "address", None),
        "date_of_birth": u.date_of_birth.isoformat() if getattr(u, "date_of_birth", None) else None,
        "last_active": u.last_active.isoformat() if getattr(u, "last_active", None) else None,
        "has_audio_embedding": bool(getattr(u, "audio_embedding", None)),
    }


def serialize_account(a: Account) -> Dict[str, Any]:
    return {
        "account_id": str(a.account_id),
        "account_number": a.account_number,
        "user_id": str(a.user_id),
        "account_type": a.account_type,
        "currency": a.currency,
        "balance": float(a.balance) if a.balance is not None else 0.0,
        "available_balance": float(a.available_balance) if a.available_balance is not None else None,
        "status": a.status,
        "interest_rate": float(a.interest_rate) if getattr(a, "interest_rate", None) is not None else None,
        "overdraft_limit": float(a.overdraft_limit) if getattr(a, "overdraft_limit", None) is not None else None,
        "opened_at": a.opened_at.isoformat() if getattr(a, "opened_at", None) else None,
        "closed_at": a.closed_at.isoformat() if getattr(a, "closed_at", None) else None,
        "created_at": a.created_at.isoformat() if getattr(a, "created_at", None) else None,
        "updated_at": a.updated_at.isoformat() if getattr(a, "updated_at", None) else None,
    }


def serialize_tx(t: Transaction) -> Dict[str, Any]:
    return {
        "transaction_id": str(t.transaction_id),
        "transaction_reference": t.transaction_reference,
        "account_id": str(t.account_id),
        "entry_type": t.entry_type,
        "transaction_type": t.transaction_type,
        "amount": float(t.amount) if t.amount is not None else None,
        "currency": t.currency,
        "status": t.status,
        "description": t.description,
        "fee": float(t.fee) if getattr(t, "fee", None) is not None else None,
        "balance_after": float(t.balance_after) if getattr(t, "balance_after", None) is not None else None,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if getattr(t, "updated_at", None) else None,
        "completed_at": t.completed_at.isoformat() if getattr(t, "completed_at", None) else None,
    }


def serialize_beneficiary(b: Beneficiary) -> Dict[str, Any]:
    return {
        "beneficiary_id": str(b.beneficiary_id),
        "user_id": str(b.user_id),
        "nickname": b.nickname,
        "account_number": b.beneficiary_account_number,
        "bank_name": b.bank_name,
        "is_internal": bool(b.is_internal),
    }


def serialize_card(c: Card) -> Dict[str, Any]:
    return {
        "card_id": str(c.card_id),
        "user_id": str(c.user_id),
        "account_id": str(c.account_id) if getattr(c, "account_id", None) else None,
        "card_number": c.card_number,
        "card_type": c.card_type,
        "network": c.network,
        "last4": c.last4,
        "credit_limit": float(c.credit_limit) if c.credit_limit is not None else None,
        "current_due": float(c.current_due) if c.current_due is not None else None,
        "min_due": float(c.min_due) if c.min_due is not None else None,
        "due_date": c.due_date.isoformat() if getattr(c, "due_date", None) else None,
        "status": c.status,
        "created_at": c.created_at.isoformat() if getattr(c, "created_at", None) else None,
        "updated_at": c.updated_at.isoformat() if getattr(c, "updated_at", None) else None,
    }


def serialize_loan(l: Loan) -> Dict[str, Any]:
    return {
        "loan_id": str(l.loan_id),
        "user_id": str(l.user_id),
        "loan_type": l.loan_type,
        "principal_amount": float(l.principal_amount) if l.principal_amount is not None else None,
        "outstanding_amount": float(l.outstanding_amount) if l.outstanding_amount is not None else None,
        "interest_rate": float(l.interest_rate) if l.interest_rate is not None else None,
        "emi_amount": float(l.emi_amount) if l.emi_amount is not None else None,
        "emi_day_of_month": l.emi_day_of_month,
        "next_due_date": l.next_due_date.isoformat() if getattr(l, "next_due_date", None) else None,
        "status": l.status,
        "created_at": l.created_at.isoformat() if getattr(l, "created_at", None) else None,
        "updated_at": l.updated_at.isoformat() if getattr(l, "updated_at", None) else None,
    }


def serialize_reminder(r: Reminder) -> Dict[str, Any]:
    return {
        "reminder_id": str(r.reminder_id),
        "user_id": str(r.user_id),
        "reminder_type": r.reminder_type,
        "title": r.title,
        "description": r.description,
        "due_date": r.due_date.isoformat() if getattr(r, "due_date", None) else None,
        "linked_loan_id": str(r.linked_loan_id) if getattr(r, "linked_loan_id", None) else None,
        "linked_card_id": str(r.linked_card_id) if getattr(r, "linked_card_id", None) else None,
        "status": r.status,
        "created_at": r.created_at.isoformat() if getattr(r, "created_at", None) else None,
        "updated_at": r.updated_at.isoformat() if getattr(r, "updated_at", None) else None,
    }
