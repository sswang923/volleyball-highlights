#!/usr/bin/env python3
"""Group DJI files into games via filename START timestamp + duration.
end = start + duration; next file within GAP_TOL == auto-split (same game).
Only games on/after MIN_DATE are shown. Read-only: no stitching."""
import re, json, subprocess
from datetime import datetime, timedelta
from pathlib import Path

VIDEO_DIR = Path("/mnt/d/volleyball")
GAP_TOL = 120
MIN_DATE = "20260619"
DJI_RE = re.compile(r"DJI_(\d{14})_(\d+)", re.IGNORECASE)

def dur(p):
    try:
        r=subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_format",str(p)],
                         capture_output=True,text=True,check=True)
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception: return 0.0

def hms(s): s=int(s); return f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}"

files=sorted([f for f in VIDEO_DIR.iterdir()
              if f.suffix.upper() in (".MP4",".MOV") and not f.name.startswith("._")],
             key=lambda f:f.name)
rows=[]
for f in files:
    m=DJI_RE.match(f.name)
    if not m: continue
    st=datetime.strptime(m.group(1),"%Y%m%d%H%M%S"); d=dur(f)
    rows.append({"seq":int(m.group(2)),"f":f,"start":st,"dur":d,
                 "end":st+timedelta(seconds=d),"date":m.group(1)[:8]})

games=[]; cur=None
for r in rows:
    new = cur is None or r["date"]!=cur[-1]["date"] or (r["start"]-cur[-1]["end"]).total_seconds()>GAP_TOL
    if new: cur=[]; games.append(cur)
    cur.append(r)

games=[g for g in games if g[0]["date"]>=MIN_DATE and sum(x["dur"] for x in g)>0]

print(f"{'GAME':<6}{'DATE':<11}{'START':<9}{'END':<9}{'DUR':>9}{'SIZE':>9}   SEGMENTS")
print("-"*78)
for i,g in enumerate(games,1):
    sz=sum(r['f'].stat().st_size for r in g)/1e9
    d=sum(r['dur'] for r in g)
    dt=datetime.strptime(g[0]['date'],"%Y%m%d").strftime("%b %d")
    seg=", ".join(f"{r['seq']:04d}" for r in g)
    print(f"{i:<6}{dt:<11}{g[0]['start'].strftime('%H:%M:%S'):<9}"
          f"{g[-1]['end'].strftime('%H:%M:%S'):<9}{hms(d):>9}{sz:>7.1f}GB   [{seg}]")
print("-"*78)
print(f"{len(games)} games (>= {MIN_DATE[:4]}-{MIN_DATE[4:6]}-{MIN_DATE[6:]}).")
