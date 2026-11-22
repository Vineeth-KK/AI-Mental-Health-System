import os
from datetime import timedelta
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    # 1. Secret Key Configuration (stable secret key, never purely random in memory)
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY') or 'careai-mental-health-system-stable-key-2026'

    # 2. Environment Detection
    is_cloud_env = bool(os.getenv('RENDER') or os.getenv('PORT') or os.getenv('DYNO'))

    # 3. Database Architecture (PostgreSQL in production, canonical SQLite locally)
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        # SQLAlchemy 1.4+ / 2.0+ requires 'postgresql://' instead of legacy 'postgres://'
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        SQLALCHEMY_DATABASE_URI = db_url
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "pool_size": 10,
            "max_overflow": 20
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

    # 5. Multi-Agent & Ollama Settings
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
    configured_host = os.getenv('OLLAMA_HOST', '').strip()

    if is_cloud_env and (not configured_host or '127.0.0.1' in configured_host or 'localhost' in configured_host or 'render-tyler-time-adsl' in configured_host):
        OLLAMA_HOST = 'https://fence-ross-spotlight-remains.trycloudflare.com'
    else:
        OLLAMA_HOST = configured_host or 'http://127.0.0.1:11434'

    CHROMA_DB_PATH = os.path.join(basedir, "instance", "chroma_db")
