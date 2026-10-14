#!/usr/bin/env python3
"""
YouTube uploader — uploads a video as Unlisted to your personal channel.

First run: opens a browser for Google login and saves a token locally.
Subsequent runs: uses the saved token (no browser needed).

Usage:
    python youtube_upload.py -v /mnt/d/volleyball/game_20260320/stitched_20260320_game01.mp4
    python youtube_upload.py -v /path/to/video.mp4 --title "My Game" --description "Game notes"
"""

import argparse
import os
import pickle
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRETS = Path(__file__).parent / "client_secrets.json"
TOKEN_FILE     = Path(__file__).parent / "youtube_token.pkl"


def get_authenticated_service():
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import googleapiclient.discovery

    creds = None
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)
        print("[+] Token saved — future uploads won't need browser login")

    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)


def upload(video_path: Path, title: str, description: str):
    from googleapiclient.http import MediaFileUpload

    youtube = get_authenticated_service()

    print(f"[+] Uploading: {video_path.name}  ({video_path.stat().st_size / 1e9:.1f} GB)")
    print("    This will take a while depending on your upload speed ...")

    body = {
        "snippet": {
            "title":       title,
            "description": description,
            "categoryId":  "17",   # Sports
        },
        "status": {
            "privacyStatus": "unlisted",
        },
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10 MB chunks
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"    Upload progress: {pct}%", end="\r")

    video_id  = response["id"]
    video_url = f"https://youtu.be/{video_id}"

    print(f"\n[+] Upload complete!")
    print(f"    URL (unlisted): {video_url}")
    print(f"\n    Copy this link into WhatsApp:\n    {video_url}\n")
    return video_url


def main():
    parser = argparse.ArgumentParser(description="Upload a video to YouTube as Unlisted")
    parser.add_argument("-v", "--video",       required=True, help="Path to MP4 file")
    parser.add_argument("--title",             default=None,  help="Video title (default: filename)")
    parser.add_argument("--description",       default="",    help="Video description")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    title = args.title
    if not title:
        date_str = video_path.stem  # fallback
        team = input("Team name (e.g. Aragon JV): ").strip() or "Unknown"
        desc = input("Game description (optional, press Enter to skip): ").strip()
        title = f"{team} — {date_str}"
        if not args.description:
            args.description = desc

    upload(video_path, title, args.description)


if __name__ == "__main__":
    main()
