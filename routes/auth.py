# routes/auth.py
"""Authentication routes: registration, login, logout."""

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db, login_manager
from models.user import User
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError

auth_bp = Blueprint("auth", __name__)

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
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
    confirm = PasswordField("Confirm Password", validators=[DataRequired(), EqualTo("password")])
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
        clean_username = form.username.data.strip()
        clean_email = form.email.data.strip().lower()
        user = User(
            username=clean_username,
            email=clean_email,
            role="user",
            is_admin=False
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful! You can now log in with your credentials.", "success")
        return redirect(url_for("auth.login"))
    return render_template("auth/register.html", form=form)

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.dashboard"))
    form = LoginForm()
    if form.validate_on_submit():
        raw_email = (form.email.data or "").strip()
        clean_email = raw_email.lower()
        raw_password = form.password.data or ""

        # Case-insensitive lookup for seamless user experience
        user = User.query.filter(db.func.lower(User.email) == clean_email).first()

        # If default credentials submitted on a fresh database, auto-seed
        if not user and clean_email in ["demo@example.com", "counselor@careai.com", "admin@careai.com"]:
            try:
                from seed_users import seed_default_accounts
                seed_default_accounts()
                user = User.query.filter(db.func.lower(User.email) == clean_email).first()
            except Exception:
                pass

        if user and user.check_password(raw_password):
            remember_choice = bool(form.remember.data) if hasattr(form, 'remember') and form.remember.data is not None else True
            login_user(user, remember=remember_choice)
            flash(f"Logged in successfully. Welcome back, {user.username}!", "success")
            next_page = request.args.get("next")
            if next_page and next_page.startswith("/"):
                # Non-counselors attempting to reach clinical portal get redirected to their dashboard
                if "/counselor" in next_page and not (user.is_counselor() if hasattr(user, "is_counselor") else False):
                    return redirect(url_for("dashboard.dashboard"))
                return redirect(next_page)
            return redirect(url_for("dashboard.dashboard"))
        else:
            flash("Invalid email or password. Please check your credentials.", "danger")
    return render_template("auth/login.html", form=form)

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
