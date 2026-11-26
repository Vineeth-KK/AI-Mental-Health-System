# services/db.py
"""Database engine and session helper for the Flask app."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import Config

DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URI") or Config.SQLALCHEMY_DATABASE_URI

engine = create_engine(
    DATABASE_URL,
    echo=False,
    **Config.SQLALCHEMY_ENGINE_OPTIONS
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
