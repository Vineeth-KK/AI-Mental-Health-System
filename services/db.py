# services/db.py
"""Database engine and session helper for the Flask app."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '..', '.env'))

DATABASE_URL = os.getenv(
    "SQLALCHEMY_DATABASE_URI",
    "mysql+pymysql://mhapp_user:YourSecurePassword123!@localhost:3306/mental_health_db"
)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
