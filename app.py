"""Streamlit Cloud Root Entrypoint.

Allows Streamlit Community Cloud to run from repository root (app.py)
or from subfolder (dashboard/app.py) without path resolution issues.
"""

import sys
import os

repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Run dashboard application code
dashboard_app_path = os.path.join(repo_root, "dashboard", "app.py")
with open(dashboard_app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, dashboard_app_path, "exec"))
