"""create chat audit logs

Revision ID: 0001_create_chat_audit_logs
Revises:
Create Date: 2026-05-29
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_create_chat_audit_logs"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("session_id", sa.String(length=80), nullable=False),
        sa.Column("trace_id", sa.String(length=80), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chat_audit_logs_session_id", "chat_audit_logs", ["session_id"])
    op.create_index("ix_chat_audit_logs_trace_id", "chat_audit_logs", ["trace_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_chat_audit_logs_trace_id", table_name="chat_audit_logs")
    op.drop_index("ix_chat_audit_logs_session_id", table_name="chat_audit_logs")
    op.drop_table("chat_audit_logs")
