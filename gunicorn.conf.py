# gunicorn.conf.py
"""Gunicorn production configuration for Render and cloud deployments.
Automatically binds to Render's dynamic PORT and optimizes memory for 512MB RAM tier.
"""

import os

# Automatically bind to 0.0.0.0:$PORT (Render injects $PORT, defaults to 5000 locally)
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# Memory-optimized worker configuration for 512MB container
workers = 1
threads = 2
timeout = 120
keepalive = 5

# Logging to standard output for Render Log viewer
accesslog = "-"
errorlog = "-"
loglevel = "info"
