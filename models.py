from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    submitter = Column(String, nullable=False)
    cluster = Column(String, nullable=False)
    partition = Column(String, nullable=False)
    script_content = Column(Text, nullable=False)
    estimated_hours = Column(Float, nullable=False)
    gpu_count = Column(Integer, default=1)
    description = Column(Text)

    # Status lifecycle:
    # pending → validated/invalid → approved/rejected → submitted → running → completed/failed
    # Any pre-dispatch status can become 'withdrawn'
    status = Column(String, default="pending", nullable=False)

    slurm_job_id = Column(String)
    is_valid = Column(Boolean, default=False)
    validation_message = Column(Text)
    admin_approved = Column(Boolean, default=False)
    notes = Column(Text)  # admin rejection reason or comments

    submitted_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime)
    dispatched_at = Column(DateTime)
    completed_at = Column(DateTime)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    api_key = Column(String, unique=True, nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
