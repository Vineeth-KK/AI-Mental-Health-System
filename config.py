import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
    # Choose database based on environment variables
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        SQLALCHEMY_DATABASE_URI = db_url
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
    else:
        # Local SQLite fallback for quick testing and local persistence
        db_path = os.path.join(basedir, "instance", "mhapp.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path.replace(os.sep, '/')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = True
    TEMPLATES_AUTO_RELOAD = True

    # Multi-Agent & RAG settings
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
    # If running on Render cloud and OLLAMA_HOST is not explicitly set, default to active tunnel
    if os.getenv('RENDER') and not os.getenv('OLLAMA_HOST'):
        OLLAMA_HOST = 'https://render-tyler-time-adsl.trycloudflare.com'
    else:
        OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
    CHROMA_DB_PATH = os.path.join(basedir, "instance", "chroma_db")

