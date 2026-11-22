# app.py
"""Flask application factory for the AI-Powered Mental Health Monitoring System."""

import os
import logging
from flask import Flask, redirect, url_for, jsonify
from flask_login import current_user
from sqlalchemy import text
from config import Config
from extensions import db, login_manager, csrf

logger = logging.getLogger(__name__)

def create_app(config_class=Config):
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    # Root route redirect
    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard.dashboard"))
        return redirect(url_for("auth.login"))

    # Production health-check endpoint with live database connection diagnostic
    @app.route("/health", methods=["GET"])
    def health():
        db_status = "connected"
        db_type = "unknown"
        try:
            db.session.execute(text("SELECT 1"))
            raw_url = str(db.engine.url)
            if "postgresql" in raw_url:
                db_type = "postgresql"
            elif "mysql" in raw_url:
                db_type = "mysql"
            elif "sqlite" in raw_url:
                db_type = "sqlite"
        except Exception as e:
            db_status = f"error: {str(e)[:80]}"
            logger.error(f"Health check database ping failed: {e}")

        return jsonify({
            "status": "healthy" if db_status == "connected" else "degraded",
            "service": "AI Mental Health Monitoring System",
            "database": {
                "status": db_status,
                "type": db_type
            },
            "multi_agent_system": "operational",
            "rag_service": "active",
            "ollama_host": app.config.get("OLLAMA_HOST"),
            "ollama_model": app.config.get("OLLAMA_MODEL"),
            "version": "v2.4-persistent-db"
        }), 200

    # Import and register blueprints
    from routes.auth import auth_bp
    from routes.checkin import checkin_bp
    from routes.journal import journal_bp
    from routes.dashboard import dashboard_bp
    from routes.chat import chat_bp
    from routes.counselor import counselor_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(checkin_bp)
    app.register_blueprint(journal_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(counselor_bp)

    # Create database tables and run safe additive schema migrations
    from services.db_migration import run_additive_migrations
    run_additive_migrations(app)

    return app
