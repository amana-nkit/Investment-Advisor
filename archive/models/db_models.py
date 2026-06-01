"""
models/db_models.py
SQLAlchemy ORM models backing the L3 (long-term structured) memory layer.
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, Integer, JSON, DateTime, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session
from core.config import get_settings


class Base(DeclarativeBase):
    pass


class CustomerProfile(Base):
    __tablename__ = "customer_profiles"

    id                       = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name                     = Column(String, nullable=False)
    email                    = Column(String, unique=True, nullable=False)
    age                      = Column(Integer)
    annual_income_inr        = Column(Float)
    risk_appetite            = Column(String)
    investment_horizon_years = Column(Integer)
    financial_goals          = Column(JSON, default=list)
    created_at               = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at               = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))


class ProposalRecord(Base):
    __tablename__ = "proposals"

    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id   = Column(String, nullable=False, index=True)
    proposal_json = Column(JSON)
    proposal_text = Column(Text)        # plain text for embedding into L4
    pdf_path      = Column(String)
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ConsentRecord(Base):
    """DPDP Act (India) — explicit consent log."""
    __tablename__ = "consent_records"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String, nullable=False, index=True)
    granted_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ip_address  = Column(String)
    purpose     = Column(String, default="investment_advisory")


def get_engine():
    settings = get_settings()
    return create_engine(settings.database_url, echo=False)


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine
