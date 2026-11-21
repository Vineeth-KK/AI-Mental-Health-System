# app.py
"""Flask application factory for the AI-Powered Mental Health Monitoring System."""

import os
from flask import Flask, redirect, url_for, jsonify
from flask_login import current_user
from config import Config
from extensions import db, login_manager, csrf

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

    # Production health-check endpoint
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "service": "AI Mental Health Monitoring System",
            "multi_agent_system": "operational",
            "rag_service": "active"
        }), 200

    # Import and register blueprints
    from routes.auth import auth_bp
    from routes.checkin import checkin_bp
    from routes.journal import journal_bp
    from routes.dashboard import dashboard_bp
    from routes.chat import chat_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(checkin_bp)
    app.register_blueprint(journal_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(chat_bp)

    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()

    return app
