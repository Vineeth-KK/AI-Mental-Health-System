import os
from datetime import timedelta
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    # 1. Secret Key Configuration (stable secret key, never purely random in memory)
    SECRET_KEY = (os.getenv('FLASK_SECRET_KEY') or os.getenv('SECRET_KEY') or 'careai-mental-health-system-stable-key-2026').strip().strip("'\"")

    # 2. Environment Detection
    is_cloud_env = bool(os.getenv('RENDER') or os.getenv('PORT') or os.getenv('DYNO'))

    # 3. Database Architecture (PostgreSQL in production, canonical SQLite locally)
    db_url = (os.getenv('DATABASE_URL') or '').strip().strip("'\"")
    if db_url:
        # SQLAlchemy 2.1+ defaults 'postgresql://' to psycopg (v3).
        # Normalize legacy 'postgres://' or driverless 'postgresql://' to 'postgresql+psycopg2://'
        # to guarantee compatibility with psycopg2-binary while psycopg 3 is also supported.
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        SQLALCHEMY_DATABASE_URI = db_url
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "pool_size": 5,
            "max_overflow": 10
        }
    elif all([
        os.getenv('MYSQL_USER'),
        os.getenv('MYSQL_PASSWORD'),
        os.getenv('MYSQL_HOST'),
        os.getenv('MYSQL_DATABASE')
    ]):
        SQLALCHEMY_DATABASE_URI = (
            f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@"
            f"{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT', '3306')}/"
            f"{os.getenv('MYSQL_DATABASE')}"
        )
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_pre_ping": True,
            "pool_recycle": 3600
        }
    else:
        # Local canonical SQLite database: absolute path guaranteed
        CANONICAL_DB_PATH = os.path.join(basedir, "instance", "mhapp.db")
        os.makedirs(os.path.dirname(CANONICAL_DB_PATH), exist_ok=True)
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{CANONICAL_DB_PATH.replace(os.sep, '/')}"
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_pre_ping": True
        }

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = True
    TEMPLATES_AUTO_RELOAD = True

    # 4. Session & Cookie Persistence Settings (14-day persistent sessions)
    SESSION_COOKIE_NAME = 'careai_session'
    PERMANENT_SESSION_LIFETIME = timedelta(days=14)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_SECURE = bool(is_cloud_env)  # Secure HTTPS in production, False for local HTTP
    REMEMBER_COOKIE_DURATION = timedelta(days=14)
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = 'Lax'
    REMEMBER_COOKIE_SECURE = bool(is_cloud_env)

    # 5. Environment & Dual AI Provider Architecture (Ollama local / Groq cloud)
    APP_ENV = os.getenv('APP_ENV', 'production' if is_cloud_env else 'development').strip().lower()
    AI_PROVIDER = os.getenv('AI_PROVIDER', 'groq' if is_cloud_env else 'ollama').strip().lower()

    # Local Ollama Settings (default: http://127.0.0.1:11434 with llama3.2)
    OLLAMA_BASE_URL = (os.getenv('OLLAMA_BASE_URL') or os.getenv('OLLAMA_HOST') or 'http://127.0.0.1:11434').rstrip('/')
    OLLAMA_HOST = OLLAMA_BASE_URL  # Backward compatibility
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
    OLLAMA_TIMEOUT = float(os.getenv('OLLAMA_TIMEOUT', '30.0'))

    # Hosted Groq Cloud Settings (production Render inference)
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '').strip().strip("'\"")
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant').strip()
    GROQ_BASE_URL = (os.getenv('GROQ_BASE_URL') or 'https://api.groq.com/openai/v1').rstrip('/')
    GROQ_TIMEOUT = float(os.getenv('GROQ_TIMEOUT', '30.0'))

    CHROMA_DB_PATH = os.path.join(basedir, "instance", "chroma_db")
