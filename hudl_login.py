#!/usr/bin/env python3
"""Bootstrap a Hudl login session.

Opens a headed Chromium with a PERSISTENT profile (./hudl_profile) so you can
log in manually — including SSO / 2FA. The cookies + session are saved to that
profile dir and reused by hudl_upload.py, so you only log in once.

Run:  python3 hudl_login.py
Then: log in to Hudl in the window that opens. When you've reached your
      dashboard, just close the browser window — the session is saved.
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROFILE = Path(__file__).parent / "hudl_profile"
LOGIN_URL = "https://www.hudl.com/login"


def main():
    PROFILE.mkdir(exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE),
            headless=False,
            no_viewport=True,
            args=["--start-maximized"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(LOGIN_URL)
        print(">> Log in to Hudl in the browser window (SSO/2FA is fine).")
        print(">> When you reach your dashboard, CLOSE the window to save the session.")
        # Stay alive until the user closes the browser (all pages gone).
        try:
            while ctx.pages:
                time.sleep(2)
        except Exception:
            pass
        try:
            ctx.close()
        except Exception:
            pass
    print(f">> Session saved to {PROFILE}")


if __name__ == "__main__":
    main()
