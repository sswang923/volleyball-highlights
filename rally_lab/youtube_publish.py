#!/usr/bin/env python3
"""
Put rally chapters and a rally caption track on an uploaded YouTube video.

Needs the youtube.force-ssl scope (edit descriptions, captions), which the
upload token (youtube_token.pkl, upload-only) does not have, so this keeps its
own token: youtube_token_manage.pkl. Authorize once — you sign in yourself:

    python3 rally_lab/youtube_publish.py auth          # prints a URL to open in your browser

Then:
    python3 rally_lab/youtube_publish.py find --query "Oct 4 2026 — Game 2"
    python3 rally_lab/youtube_publish.py publish --key 20261004_game02 --video VIDEO_ID           # dry run
    python3 rally_lab/youtube_publish.py publish --key 20261004_game02 --video VIDEO_ID --apply   # writes

The description keeps everything outside the "▶ Rallies … ◀ end rallies" block;
re-publishing replaces only that block. The caption track is named "Rallies".
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import youtube_chapters as yc  # noqa: E402

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
CLIENT_SECRETS = REPO / "client_secrets.json"
TOKEN_FILE = REPO / "youtube_token_manage.pkl"
PORT = 8091
CAPTION_NAME = "Rallies"


def out(d):
    print(json.dumps(d, ensure_ascii=False))


def service():
    from google.auth.transport.requests import Request
    import googleapiclient.discovery

    if not TOKEN_FILE.exists():
        raise SystemExit(json.dumps({"error": "not authorized for editing — run: python3 rally_lab/youtube_publish.py auth"}))
    with open(TOKEN_FILE, "rb") as f:
        creds = pickle.load(f)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)
    return googleapiclient.discovery.build("youtube", "v3", credentials=creds, cache_discovery=False)


def cmd_auth(_):
    """Same local-callback flow as auth_youtube.py, edit scope, separate token file."""
    import wsgiref.simple_server
    from urllib.parse import parse_qs
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
    flow.redirect_uri = f"http://localhost:{PORT}/"
    url, _ = flow.authorization_url(access_type="offline", prompt="select_account consent")
    print(f"\nOpen this URL in your browser and allow access:\n\n  {url}\n\nWaiting on port {PORT} ...", flush=True)
    code = [None]

    class Quiet(wsgiref.simple_server.WSGIRequestHandler):
        def log_message(self, *a):
            pass

    def app(environ, start_response):
        code[0] = parse_qs(environ.get("QUERY_STRING", "")).get("code", [None])[0]
        start_response("200 OK", [("Content-Type", "text/html")])
        return [b"<h2>Done - you can close this tab.</h2>"]

    srv = wsgiref.simple_server.make_server("127.0.0.1", PORT, app, handler_class=Quiet)
    srv.handle_request()
    if not code[0]:
        raise SystemExit("No code received - try again.")
    flow.fetch_token(code=code[0])
    with open(TOKEN_FILE, "wb") as f:
        pickle.dump(flow.credentials, f)
    print("Saved edit token:", TOKEN_FILE.name)


def cmd_find(args):
    yt = service()
    ch = yt.channels().list(part="contentDetails", mine=True).execute()
    uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    found, token = [], None
    q = args.query.lower()
    for _ in range(10):
        r = yt.playlistItems().list(part="snippet", playlistId=uploads, maxResults=50, pageToken=token).execute()
        for it in r["items"]:
            s = it["snippet"]
            if q in s["title"].lower():
                found.append({"video_id": s["resourceId"]["videoId"], "title": s["title"],
                              "published": s["publishedAt"][:10]})
        token = r.get("nextPageToken")
        if not token:
            break
    out({"query": args.query, "matches": found[:20]})


def merge_description(old, block):
    if yc.BLOCK_START in old and yc.BLOCK_END in old:
        pre = old.split(yc.BLOCK_START)[0].rstrip()
        post = old.split(yc.BLOCK_END, 1)[1].strip()
    else:
        pre, post = old.rstrip(), ""
    return "\n\n".join(x for x in (pre, block, post) if x)


def cmd_publish(args):
    res = yc.build(args.key, args.source, True if args.scores else None, args.team)
    block = Path(res["chapters_file"]).read_text(encoding="utf-8").strip()
    srt = Path(res["captions_file"]).read_text(encoding="utf-8")
    yt = service()
    v = yt.videos().list(part="snippet", id=args.video).execute()
    if not v["items"]:
        raise SystemExit(json.dumps({"error": f"video {args.video} not found on your channel"}))
    sn = v["items"][0]["snippet"]
    new_desc = merge_description(sn.get("description", ""), block)
    plan = {"video": args.video, "title": sn["title"], "chapters": res["chapters"], "rallies": res["rallies"],
            "scores_shown": res["scores_shown"], "reviewed_labels": res["reviewed"],
            "description_chars": len(new_desc), "caption_track": CAPTION_NAME if args.captions else None,
            "applied": False}
    if len(new_desc) > 5000:
        plan["error"] = "description would exceed YouTube's 5000-character limit"
        return out(plan)
    if not args.apply:
        plan["note"] = "dry run - re-run with --apply to write"
        return out(plan)

    body = {"id": args.video, "snippet": {"title": sn["title"], "categoryId": sn["categoryId"],
                                          "description": new_desc, "tags": sn.get("tags", []),
                                          "defaultLanguage": sn.get("defaultLanguage")}}
    body["snippet"] = {k: v for k, v in body["snippet"].items() if v is not None}
    yt.videos().update(part="snippet", body=body).execute()

    if args.captions:
        from googleapiclient.http import MediaInMemoryUpload
        media = MediaInMemoryUpload(srt.encode("utf-8"), mimetype="application/octet-stream", resumable=False)
        tracks = yt.captions().list(part="snippet", videoId=args.video).execute().get("items", [])
        mine = [t for t in tracks if t["snippet"].get("name") == CAPTION_NAME]
        if mine:
            yt.captions().update(part="snippet", body={"id": mine[0]["id"], "snippet": mine[0]["snippet"]},
                                 media_body=media).execute()
        else:
            yt.captions().insert(part="snippet", media_body=media, body={"snippet": {
                "videoId": args.video, "language": "en", "name": CAPTION_NAME, "isDraft": False}}).execute()
    plan["applied"] = True
    plan["url"] = f"https://youtu.be/{args.video}"
    out(plan)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auth")
    f = sub.add_parser("find")
    f.add_argument("--query", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--key", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--source", choices=["auto", "event_log"], default="event_log")
    p.add_argument("--scores", action="store_true")
    p.add_argument("--team", default="Tyrants")
    p.add_argument("--no-captions", dest="captions", action="store_false")
    p.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    {"auth": cmd_auth, "find": cmd_find, "publish": cmd_publish}[a.cmd](a)


if __name__ == "__main__":
    main()
