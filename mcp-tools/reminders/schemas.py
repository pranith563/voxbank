"""
Schemas for Reminders Tool
"""

from pydantic import BaseModel
from typing import Optional


class ReminderInput(BaseModel):
    reminder_type: str  # 'payment', 'bill', 'custom'
    title: str
    description: Optional[str] = None
    due_date: str


class ReminderOutput(BaseModel):
    reminder_id: str
    status: str
    created_at: str

