"""tests/test_authentication_suite.py
Comprehensive Verification Suite for User Authentication, Registration, Session Management,
and Multi-User Data Isolation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from app import create_app
from extensions import db
from models.user import User
from models.chat import ChatMessage
from models.checkin import CheckIn
from models.journal import JournalEntry


class TestAuthenticationAndRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False

    def setUp(self):
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()

    def tearDown(self):
        # Clean up any test users created during test methods
        for email in [
            "test_new_user@careai.com",
            "isolation_user1@careai.com",
            "isolation_user2@careai.com"
        ]:
            u = User.query.filter(db.func.lower(User.email) == email).first()
            if u:
                ChatMessage.query.filter_by(user_id=u.id).delete()
                CheckIn.query.filter_by(user_id=u.id).delete()
                JournalEntry.query.filter_by(user_id=u.id).delete()
                db.session.delete(u)
        db.session.commit()
        self.app_context.pop()

    def test_01_demo_account_login(self):
        """Scenario 1: Login using the demo account."""
        self.client.get("/logout", follow_redirects=True)
        resp = self.client.post("/login", data={
            "email": "demo@example.com",
            "password": "password123"
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Dashboard", resp.data)
        self.assertIn(b"DemoUser", resp.data)
        print("  [PASS] Scenario 1: Demo account login successful.")

    def test_02_other_existing_account_login(self):
        """Scenario 2: Login using counselor and admin existing accounts."""
        # Counselor
        client_c = self.app.test_client()
        client_c.get("/logout", follow_redirects=True)
        resp_c = client_c.post("/login", data={
            "email": "counselor@careai.com",
            "password": "password123"
        }, follow_redirects=True)
        self.assertEqual(resp_c.status_code, 200)
        self.assertIn(b"CounselorSarah", resp_c.data)

        # Admin
        client_a = self.app.test_client()
        client_a.get("/logout", follow_redirects=True)
        resp_a = client_a.post("/login", data={
            "email": "admin@careai.com",
            "password": "password123"
        }, follow_redirects=True)
        self.assertEqual(resp_a.status_code, 200)
        self.assertIn(b"AdminCareAI", resp_a.data)
        print("  [PASS] Scenario 2: Other existing registered accounts login successful.")

    def test_03_register_new_account(self):
        """Scenario 3: Register a completely new account with casing and whitespace resilience."""
        client = self.app.test_client()
        client.get("/logout", follow_redirects=True)
        resp = client.post("/register", data={
            "username": " TestNewUser ",
            "email": "  Test_New_User@CareAI.com  ",
            "password": "SecurePassword123!",
            "confirm": "SecurePassword123!"
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Registration successful", resp.data)

        # Verify in DB
        u = User.query.filter_by(email="test_new_user@careai.com").first()
        self.assertIsNotNone(u)
        self.assertEqual(u.username, "TestNewUser")
        self.assertEqual(u.email, "test_new_user@careai.com")
        self.assertEqual(u.role, "user")
        self.assertTrue(u.check_password("SecurePassword123!"))
        print("  [PASS] Scenario 3: Register completely new account successful.")

    def test_04_login_newly_registered_account(self):
        """Scenario 4: Login using the newly registered account (testing case-insensitive matching)."""
        # First register
        client = self.app.test_client()
        client.get("/logout", follow_redirects=True)
        client.post("/register", data={
            "username": "TestNewUser",
            "email": "test_new_user@careai.com",
            "password": "SecurePassword123!",
            "confirm": "SecurePassword123!"
        }, follow_redirects=True)

        # Login with capitalized email and trailing space (simulating mobile autofill)
        client_new = self.app.test_client()
        client_new.get("/logout", follow_redirects=True)
        resp = client_new.post("/login", data={
            "email": " TEST_NEW_USER@CareAI.com ",
            "password": "SecurePassword123!"
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"TestNewUser", resp.data)
        self.assertIn(b"Dashboard", resp.data)
        print("  [PASS] Scenario 4: Login using newly registered account successful.")

    def test_05_logout_and_login_again(self):
        """Scenario 5: Logout and log in again with session verification."""
        # Register
        client = self.app.test_client()
        client.get("/logout", follow_redirects=True)
        client.post("/register", data={
            "username": "TestNewUser",
            "email": "test_new_user@careai.com",
            "password": "SecurePassword123!",
            "confirm": "SecurePassword123!"
        }, follow_redirects=True)

        # Login
        client.post("/login", data={
            "email": "test_new_user@careai.com",
            "password": "SecurePassword123!"
        }, follow_redirects=True)

        # Logout
        resp_logout = client.get("/logout", follow_redirects=True)
        self.assertEqual(resp_logout.status_code, 200)
        self.assertIn(b"You have been logged out", resp_logout.data)

        # Confirm protected route redirects to login
        resp_prot = client.get("/dashboard", follow_redirects=False)
        self.assertEqual(resp_prot.status_code, 302)
        self.assertIn("/login", resp_prot.headers.get("Location", ""))

        # Log back in
        resp_login_again = client.post("/login", data={
            "email": "test_new_user@careai.com",
            "password": "SecurePassword123!"
        }, follow_redirects=True)
        self.assertEqual(resp_login_again.status_code, 200)
        self.assertIn(b"TestNewUser", resp_login_again.data)
        print("  [PASS] Scenario 5: Logout and log in again successful.")

    def test_06_data_isolation_between_users(self):
        """Scenario 6: Verify that one user's private data is not visible to another user."""
        client_a = self.app.test_client()
        client_b = self.app.test_client()
        client_a.get("/logout", follow_redirects=True)
        client_b.get("/logout", follow_redirects=True)

        # Create User A
        client_a.post("/register", data={
            "username": "UserAlpha",
            "email": "isolation_user1@careai.com",
            "password": "Password123!",
            "confirm": "Password123!"
        }, follow_redirects=True)
        user_a = User.query.filter_by(email="isolation_user1@careai.com").first()

        # Create User B
        client_b.post("/register", data={
            "username": "UserBeta",
            "email": "isolation_user2@careai.com",
            "password": "Password123!",
            "confirm": "Password123!"
        }, follow_redirects=True)
        user_b = User.query.filter_by(email="isolation_user2@careai.com").first()

        # Seed confidential data for User A
        entry_a = JournalEntry(user_id=user_a.id, content="SECRET_NOTE_USER_ALPHA_ONLY")
        chat_a = ChatMessage(user_id=user_a.id, role="user", content="CONFIDENTIAL_CHAT_USER_ALPHA")
        checkin_a = CheckIn(user_id=user_a.id, mood=2, anxiety=4, stress=4, sleep_quality=3, isolation_score=8)
        db.session.add_all([entry_a, chat_a, checkin_a])
        db.session.commit()

        # Login as User B
        client_b.post("/login", data={
            "email": "isolation_user2@careai.com",
            "password": "Password123!"
        }, follow_redirects=True)

        # Check Journal page as User B
        resp_journal = client_b.get("/journal")
        self.assertNotIn(b"SECRET_NOTE_USER_ALPHA_ONLY", resp_journal.data)

        # Check Chat page as User B
        resp_chat = client_b.get("/chat")
        self.assertNotIn(b"CONFIDENTIAL_CHAT_USER_ALPHA", resp_chat.data)

        # Check API chat history as User B
        resp_hist = client_b.get("/api/chat/history")
        hist_data = resp_hist.get_json()
        self.assertEqual(len(hist_data.get("history", [])), 0)

        # Check Dashboard as User B
        resp_dash = client_b.get("/dashboard")
        self.assertIn(b"UserBeta", resp_dash.data)
        self.assertNotIn(b"UserAlpha", resp_dash.data)

        print("  [PASS] Scenario 6: Strict cross-user data isolation verified across Journal, Chat, and Dashboard.")


if __name__ == "__main__":
    unittest.main()
