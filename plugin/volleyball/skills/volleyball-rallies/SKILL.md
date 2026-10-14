---
name: volleyball-rallies
description: Turn raw DJI volleyball game footage into rally-by-rally event logs (game start, sets, rallies, point winners) and help the user review them. Use when the user mentions their volleyball footage, games, rallies, scores, stitching DJI files, the labeler, or highlights.
---

# Volleyball rally detection

The user films club volleyball (their team is **Tyrants**) with a DJI camera behind one end line.
A local pipeline on their Windows PC (WSL Ubuntu, GTX 1650 4 GB) turns footage into rally events.
Your job is to drive that pipeline through the `volleyball` tools and keep the user's reviewed labels safe.
Do the work through the tools; do not read raw logs, videos, or large JSON files into the conversation.

## Workflow

1. **Find games** — `list_games`. Raw DJI files are grouped into games by recording gaps.
   Before stitching, sanity-check the grouping with the user (warm-ups or a camera restart mid-game can split or merge games).
2. **Stitch** — `stitch_games(date="YYYYMMDD")` → job. Lossless; about 6 minutes per game.
3. **Ask for the set scores.** The final score of every set (home-away, home = Tyrants), e.g. `19-25,25-18,15-10`.
   This is the single biggest accuracy gain: it fixes each set's rally count. If the user doesn't have them, run without.
4. **Detect** — `detect_rallies(game, set_scores)` → job. A new game takes about an hour (GPU transcode, ball
   tracking, player detection); a game already analysed takes seconds. Poll `job_status(job_id)` sparingly —
   every few minutes, not in a tight loop — and tell the user which step it is on.
5. **Report** — `rally_summary(game)` for the per-set counts. Point out sets whose rally count misses the target and that
   point winners are the weakest part.
6. **Review** — `open_labeler(game)` returns a localhost URL; the user corrects rallies there
   (W / O end a rally with home / away scoring; confirmed rallies are marked confirmed).
   If the event log is empty or only an old draft, `use_auto_events(game)` loads the new detections into it first.
7. **Measure** — `evaluate_game(game)` compares detections with the user's confirmed rallies
   (recall, precision, start/end error, point-winner accuracy) and with scoreboard checkpoints when present.

## YouTube chapters and rally captions

Games are uploaded unlisted as "Tyrants 16 Blue — <date> — Game <n>". To add a rally timeline:

1. `youtube_chapters(game)` builds one chapter per rally (plus warm-up and set breaks) and a "Rallies"
   caption track. Use the reviewed event log (default). Scores are only shown when every rally is confirmed —
   auto point winners are wrong too often to publish; only pass `show_scores=True` if the user insists.
2. `youtube_find_video("<date> — Game <n>")` to get the video id.
3. `youtube_publish(game, video_id)` is a **dry run**: show the user the title, chapter count and whether
   scores are shown. Call again with `apply=True` only after the user says to publish.

If a YouTube tool says "not authorized for editing", the upload login can't edit videos. The user must
sign in once themselves: in WSL, `python3 rally_lab/youtube_publish.py auth`, open the printed URL, allow access.
Never handle their Google password or tokens.

## Ground rules

- **Never overwrite reviewed labels.** `labels/<game>/event_log_<game>.json` is the user's golden data once it has
  confirmed rallies. `use_auto_events` refuses in that case; only pass `force=True` if the user explicitly asks,
  and say a backup is kept.
- Game ids look like `20261004_game02` (date + game number that day).
- The camera is behind the near end line; in set 1 Tyrants are usually on the near side and teams switch ends
  every set. If point winners look inverted for a whole set, the side assignment is the likely cause — say so.
- Accuracy today (game 2, against the scoreboard): rally count off by ~2 points on average with set scores,
  point winners right only ~35% of the time. Be candid about this; don't present auto results as final.
- Footage shows minors. Keep it on the user's machine: do not upload frames or video to external services.

## Without the tools

If the `volleyball` tools are not available (e.g. a chat app without the connector), give the user the
equivalent commands to run in WSL from the repo folder `/mnt/c/Users/stanl/volleyball-highlights`:

```bash
python3 dji_stitch.py -i /mnt/d/volleyball --match-date 20261004
python3 rally_lab/pipeline.py detect --key 20261004_game02 --set-scores 19-25,25-18
python3 rally_lab/pipeline.py status --key 20261004_game02
python3 rally_lab/pipeline.py evaluate --key 20261004_game02
python3 label_clips.py --video /mnt/d/volleyball/stitched_20261004_game02.mp4 --host 127.0.0.1
```

Each pipeline step prints one JSON line; ask the user to paste only the final line.
