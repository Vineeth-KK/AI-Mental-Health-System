# CareAI: Dual-Environment Rollback Guide

## Checkpoint Information
- **Pre-Change Commit:** `151163f` (*fix(db): implement persistent database architecture, session stability, and robust auth*)
- **Rollback Branch:** `careai-before-dual-environment`
- **Rollback Tag:** `careai-before-dual-environment-tag`
- **Created On:** November 2026

---

## Special Rollback Trigger: `vineethvenda`

If the user sends the single exact message:
```text
vineethvenda
```

The AI assistant must immediately execute the following rollback protocol:

1. **Halt Ongoing Edits:** Stop any further modification tasks.
2. **Verify Checkpoint:** Ensure git checkpoint branch `careai-before-dual-environment` and tag `careai-before-dual-environment-tag` point to commit `151163f`.
3. **Preserve User Data:**
   - Do NOT run destructive commands like `git clean -fd` or delete `instance/mhapp.db`.
   - Ensure `instance/`, `.env`, and all user account data are left intact.
4. **Restore Tracked Files:**
   - Restore modified code files from commit `151163f`:
     ```bash
     git checkout careai-before-dual-environment -- config.py render.yaml requirements.txt .env.example agents/ services/ routes/ tests/
     ```
   - Revert back to the single Ollama/tunnel architecture as it was configured at commit `151163f`.
5. **Verify System Integrity:**
   - Run tests: `pytest tests/`
   - Confirm local Ollama setup functions as previously configured.
6. **Report Confirmation:**
   - Report exactly which files were restored, confirm that all database tables and user data were preserved, and show the clean test status.

---

## Manual Rollback Instructions (Without AI)

If you ever need to manually roll back to the pre-dual-environment version:

1. Ensure your local database (`instance/mhapp.db`) is backed up:
   ```powershell
   Copy-Item instance/mhapp.db instance/mhapp.db.manual_backup
   ```
2. Checkout the pre-change files on your branch:
   ```powershell
   git checkout careai-before-dual-environment -- config.py render.yaml requirements.txt .env.example agents/ services/ routes/ tests/
   ```
   Or switch entirely to the checkpoint branch:
   ```powershell
   git checkout careai-before-dual-environment
   ```
3. Re-install original dependencies if needed:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run tests to confirm original behavior:
   ```powershell
   pytest tests/
   ```
