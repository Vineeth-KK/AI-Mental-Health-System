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

    # Configure ProxyFix for reverse proxy environments (e.g. Render HTTPS termination)
    if app.config.get("is_cloud_env") or os.getenv("RENDER") or os.getenv("BEHIND_PROXY", "").lower() == "true":
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)
        logger.info("ProxyFix middleware enabled for cloud reverse proxy.")

    # Root route redirect
    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard.dashboard"))
        return redirect(url_for("auth.login"))

    # Production health-check endpoint with live database connection diagnostic
    @app.route("/health", methods=["GET"])
    def health():
        from flask import request
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

        from services.ai_provider import ai_provider
        check_live = request.args.get("check_provider") in ["1", "true"]
        ai_status = ai_provider.get_status(check_live=check_live)

        return jsonify({
            "status": "healthy" if db_status == "connected" else "degraded",
            "service": "AI Mental Health Monitoring System",
            "app_env": app.config.get("APP_ENV", "development"),
            "database": {
                "status": db_status,
                "type": db_type
            },
            "ai_provider": ai_status,
            "multi_agent_system": "operational",
            "rag_service": "active",
            "version": "v2.6-prod-ready"
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
