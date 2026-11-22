# routes/auth.py
"""Authentication routes: registration, login, logout with reliable persistence."""

import logging
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db, login_manager
from models.user import User
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError

logger = logging.getLogger(__name__)
auth_bp = Blueprint("auth", __name__)

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {e}")
        return None

class RegistrationForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=25)],
        filters=[lambda x: x.strip() if x else ""]
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Email()],
        filters=[lambda x: x.strip().lower() if x else ""]
    )
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    confirm = PasswordField("Confirm Password", validators=[DataRequired(), EqualTo("password", message="Passwords must match")])
    submit = SubmitField("Register")

    def validate_username(self, field):
        clean_name = (field.data or "").strip()
        if len(clean_name) < 3:
            raise ValidationError("Username must be at least 3 characters.")
        if User.query.filter(db.func.lower(User.username) == clean_name.lower()).first():
            raise ValidationError("Username already taken. Please choose another.")

    def validate_email(self, field):
        clean_email = (field.data or "").strip().lower()
        if User.query.filter(db.func.lower(User.email) == clean_email).first():
            raise ValidationError("An account with this email already exists. Please log in.")

class LoginForm(FlaskForm):
    email = StringField(
        "Email",
        validators=[DataRequired(), Email()],
        filters=[lambda x: x.strip().lower() if x else ""]
    )
    password = PasswordField("Password", validators=[DataRequired()])
    remember = BooleanField("Remember Me", default=True)
    submit = SubmitField("Login")

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.dashboard"))
    form = RegistrationForm()
    if form.validate_on_submit():
        clean_email = (form.email.data or "").strip().lower()
        clean_username = (form.username.data or "").strip()
        try:
            user = User(
                username=clean_username,
                email=clean_email,
                role="user"
            )
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            logger.info(f"User registered successfully: {clean_username} ({clean_email})")
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("auth.login"))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration database error for {clean_email}: {e}", exc_info=True)
            flash("A database error occurred during registration. Please try again.", "danger")
    return render_template("auth/register.html", form=form)

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.dashboard"))
    form = LoginForm()
    if form.validate_on_submit():
        clean_email = (form.email.data or "").strip().lower()
        try:
            user = User.query.filter(db.func.lower(User.email) == clean_email).first()
            
            # Fallback auto-seed only for demo accounts if DB is empty
            if not user and clean_email in ["demo@example.com", "counselor@careai.com", "admin@careai.com"]:
                try:
                    from seed_users import seed_default_accounts
                    seed_default_accounts()
                    user = User.query.filter(db.func.lower(User.email) == clean_email).first()
                except Exception as seed_err:
                    logger.warning(f"Demo auto-seed failed: {seed_err}")

            if user and user.check_password(form.password.data):
                # Persistent login session: remember=True prevents logout on browser close
                remember_val = bool(form.remember.data) if hasattr(form, "remember") else True
                login_user(user, remember=remember_val)
                logger.info(f"User logged in successfully: {user.username} ({user.email})")
                flash("Logged in successfully.", "success")
                next_page = request.args.get("next")
                if not next_page or not next_page.startswith("/") or next_page.startswith("//"):
                    next_page = url_for("dashboard.dashboard")
                return redirect(next_page)
            else:
                flash("Invalid email or password.", "danger")
        except Exception as e:
            logger.error(f"Login database error: {e}", exc_info=True)
            flash("Database connection error. Please try again.", "danger")
    return render_template("auth/login.html", form=form)

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
