#!/usr/bin/env python3
"""
Starts a local OAuth callback server, prints the auth URL, waits for Chrome to redirect back.
Run: python3 auth_youtube.py
Then open the printed URL in Chrome. After clicking Allow, token is saved automatically.
"""
import pickle
import warnings
import wsgiref.simple_server
import wsgiref.util
from pathlib import Path
from urllib.parse import urlparse, parse_qs

warnings.filterwarnings("ignore")

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRETS = Path(__file__).parent / "client_secrets.json"
TOKEN_FILE = Path(__file__).parent / "youtube_token.pkl"
PORT = 8090

flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
flow.redirect_uri = f"http://localhost:{PORT}/"

auth_url, _ = flow.authorization_url(access_type="offline", prompt="select_account consent")

print(f"\n{'='*60}")
print(f"  Open this URL in Chrome:")
print(f"{'='*60}")
print(f"\n  {auth_url}\n")
print(f"{'='*60}")
print(f"\n  Waiting for Google to redirect back (port {PORT})...\n")

# Capture the redirect from Chrome
auth_code = [None]
auth_state = [None]

class _Handler(wsgiref.simple_server.WSGIRequestHandler):
    def log_message(self, *args): pass

def app(environ, start_response):
    params = parse_qs(environ.get("QUERY_STRING", ""))
    auth_code[0] = params.get("code", [None])[0]
    auth_state[0] = params.get("state", [None])[0]
    start_response("200 OK", [("Content-Type", "text/html")])
    return [b"<h2>Auth complete! You can close this tab.</h2>"]

server = wsgiref.simple_server.make_server("0.0.0.0", PORT, app, handler_class=_Handler)
server.handle_request()  # blocks until one request comes in

if not auth_code[0]:
    print("[!] No code received. Try again.")
    raise SystemExit(1)

flow.fetch_token(code=auth_code[0])
with open(TOKEN_FILE, "wb") as f:
    pickle.dump(flow.credentials, f)

print("[+] Token saved! You can now upload.")
