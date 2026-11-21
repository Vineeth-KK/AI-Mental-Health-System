# seed_users.py
from extensions import db
from models.user import User

def seed_default_accounts():
    """Ensure default demo accounts exist with correct passwords."""
    # 1. Patient Demo
    u1 = User.query.filter_by(email="demo@example.com").first()
    if not u1:
        u1 = User(username="DemoUser", email="demo@example.com", role="user")
        u1.set_password("password123")
        db.session.add(u1)
    else:
        u1.set_password("password123")
        u1.role = "user"

    # 2. Counselor Demo
    u2 = User.query.filter_by(email="counselor@careai.com").first()
    if not u2:
        u2 = User(username="CounselorSarah", email="counselor@careai.com", role="counselor")
        u2.set_password("password123")
        db.session.add(u2)
    else:
        u2.set_password("password123")
        u2.role = "counselor"

    # 3. Admin Demo
    u3 = User.query.filter_by(email="admin@careai.com").first()
    if not u3:
        u3 = User(username="AdminCareAI", email="admin@careai.com", role="admin", is_admin=True)
        u3.set_password("password123")
        db.session.add(u3)
    else:
        u3.set_password("password123")
        u3.role = "admin"
        u3.is_admin = True

    db.session.commit()

if __name__ == "__main__":
    from app import create_app
    app = create_app()
    with app.app_context():
        seed_default_accounts()
        print("\nVerification of users in DB:")
        for u in User.query.all():
            print(f"ID={u.id} | Email={u.email} | Role={u.role} | PasswordVerified={u.check_password('password123')}")
