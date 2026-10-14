// Event-Log builder — stamp game events with hotkeys while playing back.
//
// Flow:
//   GET /api/events      → { events: [...] }
//   PUT /api/events      → { events: [...] }  (full overwrite, server sorts + reseqs)

(() => {
  const video = document.getElementById("player");
  const list = document.getElementById("event-list");
  const setPill = document.getElementById("set-pill");
  const pointPill = document.getElementById("point-pill");
  const statePill = document.getElementById("state-pill");
  const statusText = document.getElementById("status-text");
  const rateSelectEl = document.getElementById("rate-select");
  const nowEl  = document.getElementById("now-time");
  const tlEl     = document.getElementById("timeline");
  const tlTrack  = document.getElementById("timeline-track");
  const tlAxis   = document.getElementById("timeline-axis");
  const tlCursor = document.getElementById("timeline-cursor");
  const tlRange  = document.getElementById("tl-range");
  const scorePill = document.getElementById("score-pill");

  const RATES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
  const FPS = 30;
  const DURATION = Math.max(1, window.DURATION_S || 1);
  const MIN_SPAN = 5;  // 5 seconds minimum visible window

  const SCOREBUG_POSITIONS = ["tl", "tc", "tr", "bl", "bc", "br", "off"];

  const state = {
    events: [],       // [{ event, t_s, set, rally_in_set, score_by, seq, ... }]
    dirty: false,
    saveTimer: null,
    rallyOnly: false,
    viewStart: 0,
    viewEnd: DURATION,
    scorebugPos: "bl",
    scorebugCustom: null,
    highlights: { collections: [] },
    activeCollectionId: null,
    activeTab: "rallies",
    activeScans: {},   // setNum -> { jobId, pct, progress, ralliesCount, lastRallyLabel }
  };

  try { state.rallyOnly = localStorage.getItem("vc.rallyOnly") === "1"; } catch (_) {}
  try {
    const sp = localStorage.getItem("vc.scorebugPos");
    if (sp && (SCOREBUG_POSITIONS.includes(sp) || sp === "custom")) state.scorebugPos = sp;
    const sc = localStorage.getItem("vc.scorebugCustom");
    if (sc) state.scorebugCustom = JSON.parse(sc);
  } catch (_) {}

  // Per-game team names
  const _gameKey = document.body.dataset.gameKey || "default";
  state.teamHome = "HOME";
  state.teamAway = "AWAY";
  try {
    state.teamHome = localStorage.getItem(`teams.${_gameKey}.home`) || "HOME";
    state.teamAway = localStorage.getItem(`teams.${_gameKey}.away`) || "AWAY";
  } catch (_) {}
  function saveTeams() {
    try {
      localStorage.setItem(`teams.${_gameKey}.home`, state.teamHome);
      localStorage.setItem(`teams.${_gameKey}.away`, state.teamAway);
    } catch (_) {}
  }
  function editTeams() {
    const h = prompt("Home team name (your team)", state.teamHome);
    if (h !== null && h.trim()) state.teamHome = h.trim().slice(0, 24);
    const a = prompt("Away team name (opponent)", state.teamAway);
    if (a !== null && a.trim()) state.teamAway = a.trim().slice(0, 24);
    saveTeams();
    applyTeamNamesToUI();
    render();
  }
  function applyTeamNamesToUI() {
    const hb = document.getElementById("btn-score-home");
    const ab = document.getElementById("btn-score-away");
    if (hb) hb.innerHTML = `+ ${escapeText(state.teamHome)} <kbd>W</kbd>`;
    if (ab) ab.innerHTML = `+ ${escapeText(state.teamAway)} <kbd>O</kbd>`;
  }

  // ── Bootstrap ───────────────────────────────────────────────────────────
  async function init() {
    const r = await fetch("/api/events");
    const data = await r.json();
    // Migrate any legacy "we"/"opp" values to "home"/"away"
    state.events = (data.events || []).map(e => {
      if (e.score_by === "we")  e.score_by = "home";
      if (e.score_by === "opp") e.score_by = "away";
      return e;
    });

    video.src = "/video";
    await loadHighlights();
    // Clean up any pre-existing out-of-order numbering before first render
    if (renumberAllSets()) {
      scheduleSave();
    }
    wireUI();
    applyTeamNamesToUI();
    applyRallyOnlyUI();
    attachScorebugDrag();
    populateVideoPicker();
    renderAxis();
    render();
    // After first render, look for any running scan jobs (e.g. survived a
    // page refresh) and resume polling so progress shows immediately.
    resumeActiveScans();
  }

  async function resumeActiveScans() {
    try {
      const r = await fetch("/api/auto-rallies/jobs");
      const data = await r.json();
      for (const job of data.jobs || []) {
        const setNum = job.set_num;
        if (setNum == null) continue;
        if (state.activeScans[setNum]) continue;   // already tracking
        state.activeScans[setNum] = {
          jobId: job.id,
          pct: job.pct || 0,
          progress: job.progress || "Resuming…",
          ralliesCount: job.rallies_count || 0,
          lastRallyLabel: job.last_rally_label || "",
          rangeLabel: "",
        };
        pollScan(setNum);
      }
      if ((data.jobs || []).length) render();
    } catch (e) {
      console.error("resume active scans failed", e);
    }
  }

  // ── Highlight collections (multi) ───────────────────────────────────────
  async function loadHighlights() {
    try {
      const r = await fetch("/api/highlights");
      const data = await r.json();
      state.highlights = data;
      if (!state.activeCollectionId && data.collections.length > 0) {
        state.activeCollectionId = data.collections[0].id;
      }
    } catch (e) { console.error("highlights load failed", e); }
  }
  function findCollection(id) {
    return state.highlights.collections.find(c => c.id === id) || null;
  }
  function collectionsContainingRally(setNum, rallyNum) {
    return state.highlights.collections.filter(c =>
      c.rallies.some(r => r.set === setNum && r.rally_in_set === rallyNum)
    );
  }
  async function apiCreateCollection(name) {
    const r = await fetch("/api/highlights", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await r.json();
    if (data.ok) {
      await loadHighlights();
      state.activeCollectionId = data.collection.id;
      render();
      return data.collection;
    }
    return null;
  }
  async function apiDeleteCollection(id) {
    await fetch(`/api/highlights/${encodeURIComponent(id)}`, { method: "DELETE" });
    await loadHighlights();
    if (state.activeCollectionId === id) {
      state.activeCollectionId = state.highlights.collections[0]?.id || null;
    }
    render();
  }
  async function apiRenameCollection(id, name) {
    await fetch(`/api/highlights/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await loadHighlights();
    render();
  }
  async function apiAddRally(collectionId, setNum, rallyNum, notes) {
    await fetch(`/api/highlights/${encodeURIComponent(collectionId)}/rallies`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ set: setNum, rally_in_set: rallyNum, notes: notes || "" }),
    });
    await loadHighlights();
    render();
  }
  async function apiRemoveRally(collectionId, setNum, rallyNum) {
    await fetch(`/api/highlights/${encodeURIComponent(collectionId)}/rallies/${setNum}/${rallyNum}`, {
      method: "DELETE",
    });
    await loadHighlights();
    render();
  }
  async function apiUpdateRallyNotes(collectionId, setNum, rallyNum, notes) {
    await fetch(`/api/highlights/${encodeURIComponent(collectionId)}/rallies/${setNum}/${rallyNum}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes }),
    });
    // No reload — let render proceed with optimistic update
  }

  async function populateVideoPicker() {
    const sel = document.getElementById("video-select");
    if (!sel) return;
    try {
      const r = await fetch("/api/videos");
      const data = await r.json();
      sel.innerHTML = "";
      for (const v of data.videos || []) {
        const opt = document.createElement("option");
        opt.value = v.game_key;          // use game_key as the value
        opt.textContent = `${v.game_key}  ·  ${v.size_mb} MB`;
        if (v.game_key === data.current_game_key) opt.selected = true;
        sel.appendChild(opt);
      }
      sel.onchange = () => {
        const u = new URL(window.location.href);
        u.searchParams.delete("video");   // strip the old param shape
        u.searchParams.set("game", sel.value);
        window.location.href = u.toString();
      };
    } catch (e) { console.error("video list failed", e); }
  }

  // Keep state.events sorted chronologically so currentSet / openRallyInSet
  // (which walk in array order) see the latest reality between saves.
  function sortEvents() {
    state.events.sort((a, b) => a.t_s - b.t_s);
  }

  // ── Save (debounced) ────────────────────────────────────────────────────
  function scheduleSave() {
    state.dirty = true;
    if (state.saveTimer) clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(saveNow, 300);
  }
  async function saveNow() {
    state.saveTimer = null;
    try {
      const r = await fetch("/api/events", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ events: state.events }),
      });
      const j = await r.json();
      if (j.ok) {
        state.events.sort((a, b) => a.t_s - b.t_s);
        state.events.forEach((e, i) => e.seq = i);
        state.dirty = false;
        render();
      }
    } catch (e) {
      console.error("save failed", e);
    }
  }

  // ── Event lookup helpers ────────────────────────────────────────────────
  function currentSet() {
    let openSet = null;
    for (const e of state.events) {
      if (e.event === "set_start") openSet = e.set;
      else if (e.event === "set_end") openSet = null;
    }
    return openSet;
  }
  function openRallyInSet(setNum) {
    let open = null;
    for (const e of state.events) {
      if (e.set !== setNum) continue;
      if (e.event === "rally_start") open = e;
      else if (e.event === "rally_end") open = null;
    }
    return open;
  }
  function highestRallyInSet(setNum) {
    let m = 0;
    for (const e of state.events) {
      if (e.set === setNum && e.event === "rally_start" && (e.rally_in_set || 0) > m) {
        m = e.rally_in_set;
      }
    }
    return m;
  }
  function highestSet() {
    let m = 0;
    for (const e of state.events) {
      if (e.event === "set_start" && (e.set || 0) > m) m = e.set;
    }
    return m;
  }

  function getRallySegments() {
    const sorted = [...state.events].sort((a, b) => a.t_s - b.t_s);
    const segs = [];
    const open = new Map();
    for (const e of sorted) {
      if (e.event === "rally_start") {
        open.set(`${e.set}/${e.rally_in_set}`, e);
      } else if (e.event === "rally_end") {
        const k = `${e.set}/${e.rally_in_set}`;
        const st = open.get(k);
        if (st) {
          segs.push({
            start: st.t_s, end: e.t_s,
            score_by: e.score_by || null,
            set: e.set, rally_in_set: e.rally_in_set,
            pending: !!(st.auto && st.confirmed === false),
          });
          open.delete(k);
        }
      }
    }
    return segs.sort((a, b) => a.start - b.start);
  }

  // Set scores from completed rally_end events
  function setScores(setNum) {
    let home = 0, away = 0;
    for (const e of state.events) {
      if (e.set !== setNum) continue;
      if (e.event === "rally_end") {
        if (e.score_by === "home") home++;
        else if (e.score_by === "away") away++;
      }
    }
    return { home, away };
  }

  // Sets won by each team (only counts sets with a set_end stamped)
  function setsWon() {
    let home = 0, away = 0;
    const setNums = new Set();
    for (const e of state.events) {
      if (e.event === "set_start") setNums.add(e.set);
    }
    for (const n of setNums) {
      const hasEnd = state.events.some(e => e.event === "set_end" && e.set === n);
      if (!hasEnd) continue;
      const { home: h, away: a } = setScores(n);
      if (h > a) home++;
      else if (a > h) away++;
    }
    return { home, away };
  }

  // ── Time-aware scoring (for scorebug sync to playhead position) ─────────
  function currentSetAtTime(t) {
    const sorted = [...state.events].sort((a, b) => a.t_s - b.t_s);
    let openSet = null;
    for (const e of sorted) {
      if (e.t_s > t) break;
      if (e.event === "set_start") openSet = e.set;
      else if (e.event === "set_end") openSet = null;
    }
    return openSet;
  }
  function setScoresAtTime(setNum, t) {
    let home = 0, away = 0;
    for (const e of state.events) {
      if (e.set !== setNum) continue;
      if (e.event === "rally_end" && e.t_s <= t) {
        if (e.score_by === "home") home++;
        else if (e.score_by === "away") away++;
      }
    }
    return { home, away };
  }
  function setsWonAtTime(t) {
    let home = 0, away = 0;
    const setNums = new Set();
    for (const e of state.events) {
      if (e.event === "set_start" && e.t_s <= t) setNums.add(e.set);
    }
    for (const n of setNums) {
      const ended = state.events.some(e => e.event === "set_end" && e.set === n && e.t_s <= t);
      if (!ended) continue;
      const { home: h, away: a } = setScoresAtTime(n, t);
      if (h > a) home++;
      else if (a > h) away++;
    }
    return { home, away };
  }

  // ── Event creation ──────────────────────────────────────────────────────
  function stamp(kind, extra = {}) {
    const overrideT = extra.t_s;
    const cleanExtra = { ...extra };
    delete cleanExtra.t_s;
    const t_s = overrideT != null
      ? +parseFloat(overrideT).toFixed(3)
      : +video.currentTime.toFixed(3);
    const e = {
      event: kind,
      t_s,
      set: null,
      rally_in_set: null,
      score_by: null,
      ...cleanExtra,
    };
    if (kind === "game_start") {
      // game_start implies no set yet; leave set null
    } else if (kind === "set_start") {
      e.set = (highestSet() || 0) + 1;
    } else if (kind === "set_end") {
      e.set = currentSet();
    } else if (kind === "rally_start") {
      const s = currentSet();
      if (!s) {
        const setStart = { event: "set_start", t_s, set: 1, rally_in_set: null, score_by: null };
        state.events.push(setStart);
      }
      e.set = currentSet() || 1;
      e.rally_in_set = highestRallyInSet(e.set) + 1;  // tentative; renumber after push
      // Auto-align any unaligned preceding game_start / set_start (for this set)
      const prev = state.events
        .filter(x => x !== e && x.t_s < t_s)
        .sort((a, b) => b.t_s - a.t_s);
      for (const ex of prev) {
        if (ex.event === "rally_start" || ex.event === "rally_end") break;
        if (ex.event === "game_start") ex.t_s = t_s;
        if (ex.event === "set_start" && ex.set === e.set) ex.t_s = t_s;
      }
    } else if (kind === "rally_end") {
      const s = currentSet();
      const open = s ? openRallyInSet(s) : null;
      if (!open) {
        flash("No rally is open. Press Space at rally start first.");
        return;
      }
      e.set = open.set;
      e.rally_in_set = open.rally_in_set;
    } else if (kind === "game_end") {
      e.set = currentSet();
    }
    state.events.push(e);
    sortEvents();
    // After a rally event, normalize the numbering and update highlights refs
    if ((kind === "rally_start" || kind === "rally_end") && e.set != null) {
      maybeRenumberRalliesInSet(e.set);
    }
    scheduleSave();
    render();
    return e;
  }

  // Walk this set's rally events chronologically. Pair each rally_start with
  // the next rally_end (regardless of their stored rally_in_set), and assign
  // 1..N to the starts in order. The matching rally_end takes the same number.
  // Returns true if anything changed; pushes a {old,new} remap to the server
  // so highlight collection refs stay valid.
  function maybeRenumberRalliesInSet(setNum) {
    const evs = state.events
      .filter(x => x.set === setNum && (x.event === "rally_start" || x.event === "rally_end"))
      .sort((a, b) => a.t_s - b.t_s);
    let counter = 0;
    let openNum = null;
    const pairs = [];
    for (const e of evs) {
      if (e.event === "rally_start") {
        counter++;
        const oldNum = e.rally_in_set;
        if (oldNum !== counter) {
          if (oldNum != null) pairs.push({ old: oldNum, new: counter });
          e.rally_in_set = counter;
        }
        openNum = counter;
      } else {  // rally_end
        if (openNum != null) {
          e.rally_in_set = openNum;
          openNum = null;
        } else {
          e.rally_in_set = null;  // orphan end with no preceding start
        }
      }
    }
    if (pairs.length > 0) {
      fetch("/api/highlights/renumber", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ set: setNum, pairs }),
      }).then(() => loadHighlights()).catch(() => {});
      return true;
    }
    return false;
  }

  // Run the renumber pass across every set that has rallies. Returns true
  // if any set had ordering corrections.
  function renumberAllSets() {
    const setNums = new Set();
    for (const e of state.events) {
      if (e.event === "rally_start" && e.set != null) setNums.add(e.set);
    }
    let changed = false;
    for (const s of setNums) {
      if (maybeRenumberRalliesInSet(s)) changed = true;
    }
    return changed;
  }

  function toggleRally() {
    const s = currentSet();
    const open = s ? openRallyInSet(s) : null;
    if (open) {
      // Split: close this rally and immediately open the next, +1ms so the
      // new start sorts strictly after the end it follows.
      const t = +video.currentTime.toFixed(3);
      const ended = stamp("rally_end", { t_s: t });
      if (ended) stamp("rally_start", { t_s: +(t + 0.001).toFixed(3) });
    } else {
      stamp("rally_start");
    }
  }

  // Convert rally_start ↔ rally_end on an existing event. Clears AI flags
  // (the user has manually edited it) and renumbers the affected set.
  function flipRallyEvent(seq) {
    const e = state.events.find(x => x.seq === seq);
    if (!e) return;
    if (e.event !== "rally_start" && e.event !== "rally_end") return;
    e.event = (e.event === "rally_start") ? "rally_end" : "rally_start";
    delete e.auto;
    delete e.confirmed;
    sortEvents();
    if (e.set != null) maybeRenumberRalliesInSet(e.set);
    scheduleSave();
    render();
  }

  // Attribute a point. Two paths:
  //   1. There's a recent rally_end at-or-before the playhead with no
  //      score_by — set its score_by. Handles "Space split, then W/O" cleanly:
  //      the previously-closed rally gets the score; the new open rally is
  //      left alone.
  //   2. Otherwise, fall back to original behavior — close the open rally
  //      with attribution.
  function scoreRally(side) {
    const ends = state.events
      .filter(e => e.event === "rally_end" && e.t_s <= video.currentTime + 0.1)
      .sort((a, b) => a.t_s - b.t_s);
    const lastEnd = ends[ends.length - 1];
    if (lastEnd && lastEnd.score_by == null) {
      lastEnd.score_by = side;
      scheduleSave();
      render();
      return;
    }
    const s = currentSet();
    const open = s ? openRallyInSet(s) : null;
    if (open) {
      stamp("rally_end", { score_by: side });
    } else {
      flash(`No rally to score. Press Space then ${side === "home" ? "W" : "O"}.`);
    }
  }

  function deleteLast() {
    if (!state.events.length) return;
    state.events.sort((a, b) => a.t_s - b.t_s);
    state.events.pop();
    renumberAllSets();
    scheduleSave();
    render();
  }
  function deleteEvent(seq) {
    state.events = state.events.filter(e => e.seq !== seq);
    renumberAllSets();
    scheduleSave();
    render();
  }
  function bumpLastTime(delta) {
    sortEvents();
    const last = state.events[state.events.length - 1];
    if (!last) return;
    last.t_s = Math.max(0, +(last.t_s + delta).toFixed(3));
    sortEvents();
    if (last.set != null && (last.event === "rally_start" || last.event === "rally_end")) {
      maybeRenumberRalliesInSet(last.set);
    }
    scheduleSave();
    render();
  }

  // Cycle the score_by on a rally_end event (we → opp → null → we …)
  function cycleScore(seq) {
    const evt = state.events.find(e => e.seq === seq);
    if (!evt || evt.event !== "rally_end") return;
    const next = evt.score_by === "home" ? "away" : evt.score_by === "away" ? null : "home";
    evt.score_by = next;
    scheduleSave();
    render();
  }

  // ── Rally-only playback ─────────────────────────────────────────────────
  function setRallyOnly(on) {
    state.rallyOnly = !!on;
    try { localStorage.setItem("vc.rallyOnly", state.rallyOnly ? "1" : "0"); } catch (_) {}
    applyRallyOnlyUI();
    if (state.rallyOnly) jumpToRallyIfNeeded();
    render();
  }
  function applyRallyOnlyUI() {
    const btn = document.getElementById("btn-rally-only");
    if (btn) btn.dataset.active = state.rallyOnly ? "true" : "false";
    if (tlEl) tlEl.dataset.rallyOnly = state.rallyOnly ? "true" : "false";
  }

  // ── Scorebug overlay ────────────────────────────────────────────────────
  function setScorebugPos(pos) {
    if (!SCOREBUG_POSITIONS.includes(pos)) return;
    state.scorebugPos = pos;
    try { localStorage.setItem("vc.scorebugPos", pos); } catch (_) {}
    renderScorebug();
  }
  function setScorebugCustom(topPct, leftPct) {
    state.scorebugPos = "custom";
    state.scorebugCustom = { topPct, leftPct };
    try {
      localStorage.setItem("vc.scorebugPos", "custom");
      localStorage.setItem("vc.scorebugCustom", JSON.stringify(state.scorebugCustom));
    } catch (_) {}
    renderScorebug();
  }
  function cycleScorebug() {
    let i = SCOREBUG_POSITIONS.indexOf(state.scorebugPos);
    if (i < 0) i = -1;  // custom → start from tl
    const next = SCOREBUG_POSITIONS[(i + 1) % SCOREBUG_POSITIONS.length];
    setScorebugPos(next);
  }
  let _lastShown = { home: null, away: null, sh: null, sa: null };
  function renderScorebug() {
    const bug = document.getElementById("scorebug");
    if (!bug) return;
    // Highlight active picker button (custom = none highlighted)
    document.querySelectorAll(".scorebug-picker .sb-pos").forEach(btn => {
      btn.dataset.active = btn.dataset.pos === state.scorebugPos ? "true" : "false";
    });

    const t = video.currentTime || 0;
    const cs = currentSetAtTime(t);
    if (!cs || state.scorebugPos === "off") {
      bug.style.display = "none";
      _lastShown = { home: null, away: null, sh: null, sa: null };
      return;
    }
    bug.style.display = "";
    bug.dataset.pos = state.scorebugPos;

    if (state.scorebugPos === "custom" && state.scorebugCustom) {
      bug.className = "scorebug pos-custom";
      bug.style.left   = `${state.scorebugCustom.leftPct}%`;
      bug.style.top    = `${state.scorebugCustom.topPct}%`;
      bug.style.right  = "auto";
      bug.style.bottom = "auto";
      bug.style.transform = "none";
    } else {
      bug.className = `scorebug pos-${state.scorebugPos}`;
      bug.style.left = "";
      bug.style.top = "";
      bug.style.right = "";
      bug.style.bottom = "";
      bug.style.transform = "";
    }

    const { home, away } = setScoresAtTime(cs, t);
    const sw = setsWonAtTime(t);
    const homeEl = bug.querySelector(".sb-home .sb-num");
    const awayEl = bug.querySelector(".sb-away .sb-num");
    const homeName = bug.querySelector(".sb-home .sb-name");
    const awayName = bug.querySelector(".sb-away .sb-name");
    const shEl   = bug.querySelector(".sb-set-num.home");
    const saEl   = bug.querySelector(".sb-set-num.away");
    if (homeName) homeName.textContent = state.teamHome;
    if (awayName) awayName.textContent = state.teamAway;

    // Flash only on score INCREASE (forward playback past a rally_end)
    if (_lastShown.home !== null && home > _lastShown.home) flashNum(homeEl);
    if (_lastShown.away !== null && away > _lastShown.away) flashNum(awayEl);

    homeEl.textContent = home;
    awayEl.textContent = away;
    shEl.textContent = sw.home;
    saEl.textContent = sw.away;
    _lastShown = { home, away, sh: sw.home, sa: sw.away };
  }
  function flashNum(el) {
    if (!el) return;
    el.classList.remove("flash");
    void el.offsetWidth;
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 750);
  }
  function attachScorebugDrag() {
    const bug = document.getElementById("scorebug");
    const stage = document.querySelector(".video-stage");
    if (!bug || !stage) return;
    bug.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const stageRect = stage.getBoundingClientRect();
      const bugRect = bug.getBoundingClientRect();
      const grabX = e.clientX - bugRect.left;
      const grabY = e.clientY - bugRect.top;
      bug.classList.add("dragging");
      function move(ev) {
        let x = ev.clientX - stageRect.left - grabX;
        let y = ev.clientY - stageRect.top  - grabY;
        x = Math.max(0, Math.min(stageRect.width  - bugRect.width,  x));
        y = Math.max(0, Math.min(stageRect.height - bugRect.height, y));
        const leftPct = (x / stageRect.width)  * 100;
        const topPct  = (y / stageRect.height) * 100;
        setScorebugCustom(topPct, leftPct);
      }
      function up() {
        bug.classList.remove("dragging");
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
      }
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
  }
  // (kept for any future explicit flashes — main flashing is driven by
  //  renderScorebug's score-increase diff against _lastShown)
  function flashScorebug(side) {
    const bug = document.getElementById("scorebug");
    if (!bug || state.scorebugPos === "off") return;
    flashNum(bug.querySelector(`.sb-${side} .sb-num`));
  }
  function jumpToRallyIfNeeded() {
    if (!state.rallyOnly) return;
    const t = video.currentTime;
    const segs = getRallySegments();
    if (!segs.length) return;
    const inSeg = segs.find(s => t >= s.start && t < s.end);
    if (inSeg) return;
    const next = segs.find(s => s.start > t);
    if (next) video.currentTime = next.start;
    else video.pause();
  }

  // ── Timeline: axis, markers, cursor, zoom ───────────────────────────────
  function clampView(start, end) {
    if (end - start < MIN_SPAN) end = start + MIN_SPAN;
    start = Math.max(0, start);
    end = Math.min(DURATION, end);
    if (end - start < MIN_SPAN) start = Math.max(0, end - MIN_SPAN);
    return [start, end];
  }
  function setView(start, end) {
    [state.viewStart, state.viewEnd] = clampView(start, end);
    renderAxis();
    render();
    updateCursor();
    updateRangeLabel();
  }
  function zoomAt(ratio, factor) {
    const span = state.viewEnd - state.viewStart;
    const t = state.viewStart + ratio * span;
    const newSpan = Math.max(MIN_SPAN, Math.min(DURATION, span * factor));
    let ns = t - ratio * newSpan;
    let ne = ns + newSpan;
    if (ns < 0) { ne -= ns; ns = 0; }
    if (ne > DURATION) { ns -= (ne - DURATION); ne = DURATION; }
    setView(ns, ne);
  }
  function zoomInAtPlayhead() {
    const span = state.viewEnd - state.viewStart;
    let ratio = (video.currentTime - state.viewStart) / span;
    if (ratio < 0 || ratio > 1) ratio = 0.5;
    zoomAt(ratio, 0.6);
  }
  function zoomOutAtPlayhead() {
    const span = state.viewEnd - state.viewStart;
    let ratio = (video.currentTime - state.viewStart) / span;
    if (ratio < 0 || ratio > 1) ratio = 0.5;
    zoomAt(ratio, 1 / 0.6);
  }
  function zoomReset() { setView(0, DURATION); }

  function pickTickInterval(span) {
    // Finer ticks when zoomed in: down to 1-second resolution
    const candidates = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
    for (const iv of candidates) {
      if (span / iv <= 12) return iv;
    }
    return 3600;
  }
  function fmtTickLabel(t, step) {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    if (step >= 600) return `${m}m`;
    if (step >= 60)  return s === 0 ? `${m}m` : `${m}:${String(s).padStart(2,"0")}`;
    return `${m}:${String(s).padStart(2,"0")}`;
  }

  function renderAxis() {
    if (!tlAxis) return;
    tlAxis.innerHTML = "";
    const span = state.viewEnd - state.viewStart;
    if (span <= 0) return;
    const step = pickTickInterval(span);
    const startTick = Math.ceil(state.viewStart / step) * step;
    for (let t = startTick; t <= state.viewEnd + 0.001; t += step) {
      const pct = ((t - state.viewStart) / span) * 100;
      const tick = document.createElement("div");
      tick.className = "tick";
      tick.style.left = `${pct}%`;
      tlAxis.appendChild(tick);

      const label = document.createElement("div");
      label.className = "tick-label";
      label.style.left = `${pct}%`;
      label.textContent = fmtTickLabel(t, step);
      tlAxis.appendChild(label);
    }
  }

  function renderTimeline(sorted) {
    if (!tlTrack) return;
    [...tlTrack.querySelectorAll(".tl-marker, .tl-span, .tl-dead")].forEach(el => el.remove());

    const span = state.viewEnd - state.viewStart;
    if (span <= 0) return;

    // Dead-time blocks (only when rally-only is on)
    if (state.rallyOnly) {
      const segs = getRallySegments();
      const drawDead = (t0, t1) => {
        if (t1 < state.viewStart || t0 > state.viewEnd) return;
        const a = Math.max(0, ((t0 - state.viewStart) / span) * 100);
        const b = Math.min(100, ((t1 - state.viewStart) / span) * 100);
        if (b - a < 0.05) return;
        const block = document.createElement("div");
        block.className = "tl-dead";
        block.style.left = `${a}%`;
        block.style.width = `${b - a}%`;
        tlTrack.appendChild(block);
      };
      let prev = 0;
      for (const s of segs) {
        if (s.start > prev) drawDead(prev, s.start);
        prev = s.end;
      }
      if (prev < DURATION) drawDead(prev, DURATION);
    }

    // Rally spans (color-coded by score_by)
    const segs2 = getRallySegments();
    for (const s of segs2) {
      if (s.end < state.viewStart || s.start > state.viewEnd) continue;
      const div = document.createElement("div");
      div.className = "tl-span" + (s.score_by ? ` ${s.score_by}` : "") + (s.pending ? " pending" : "");
      const a = Math.max(0, ((s.start - state.viewStart) / span) * 100);
      const b = Math.min(100, ((s.end - state.viewStart) / span) * 100);
      div.style.left = `${a}%`;
      div.style.width = `${Math.max(0.2, b - a)}%`;
      div.title = `Rally ${s.set}/${s.rally_in_set} · ${fmt(s.start)} → ${fmt(s.end)}`
                + (s.score_by ? ` · ${s.score_by} scored` : "");
      tlTrack.appendChild(div);
    }

    // Whistle bursts as small tick marks on the bottom of the track
    for (const e of sorted) {
      if (e.event !== "whistle") continue;
      if (e.t_s < state.viewStart || e.t_s > state.viewEnd) continue;
      const w = document.createElement("div");
      w.className = "tl-whistle";
      w.style.left = `${((e.t_s - state.viewStart) / span) * 100}%`;
      w.title = `whistle @ ${fmt(e.t_s)} (${e.duration?.toFixed?.(2) || "?"}s)`;
      tlTrack.appendChild(w);
    }

    // Markers — draggable to adjust timestamp
    for (const e of sorted) {
      if (e.event === "whistle") continue;  // drawn separately above
      if (e.t_s < state.viewStart || e.t_s > state.viewEnd) continue;
      const m = document.createElement("div");
      const pending = (e.event === "rally_start" || e.event === "rally_end") && e.auto && e.confirmed === false;
      m.className = `tl-marker ${e.event}` + (pending ? " pending" : "");
      m.style.left = `${((e.t_s - state.viewStart) / span) * 100}%`;
      m.title = `${e.event} @ ${fmt(e.t_s)}${pending ? " — AI (pending)" : ""} — drag to adjust`;
      m.dataset.seq = e.seq;
      attachMarkerDrag(m, e);
      tlTrack.appendChild(m);
    }
  }

  // Generic drag-to-adjust-time helper for sidebar rows
  function attachRowDrag(row, evt, opts = {}) {
    row.addEventListener("mousedown", (ev) => {
      // Ignore clicks on inline buttons (score-tag, del, etc.)
      if (ev.target.closest("button")) return;
      if (ev.button !== 0) return;
      ev.preventDefault();
      const startX = ev.clientX;
      const origT  = evt.t_s;
      const origT2 = opts.linkedEvent ? opts.linkedEvent.t_s : null;
      const pxPerSec = opts.pxPerSec || 30;
      let moved = false;

      const wasPlaying = !video.paused;
      if (wasPlaying) video.pause();

      const move = (e2) => {
        const dx = e2.clientX - startX;
        if (!moved && Math.abs(dx) < 4) return;
        moved = true;
        row.classList.add("dragging");
        const dt  = dx / pxPerSec;
        const newT  = Math.max(0, Math.min(DURATION, +(origT + dt).toFixed(3)));
        evt.t_s = newT;
        if (opts.linkedEvent && origT2 !== null) {
          const new2 = Math.max(0, Math.min(DURATION, +(origT2 + dt).toFixed(3)));
          opts.linkedEvent.t_s = new2;
        }
        const cell = opts.timeCell ? row.querySelector(opts.timeCell) : null;
        if (cell) cell.textContent = fmt(evt.t_s);
        try { video.currentTime = evt.t_s; } catch (_) {}
      };
      const up = (e2) => {
        row.classList.remove("dragging");
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        if (moved) {
          renumberAllSets();
          scheduleSave();
          render();
        } else {
          // Plain click — seek to the event's time
          try { video.currentTime = evt.t_s; } catch (_) {}
        }
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
  }

  function attachMarkerDrag(m, evt) {
    m.addEventListener("mousedown", (ev) => {
      if (ev.button !== 0) return;
      ev.stopPropagation();
      ev.preventDefault();
      const rect = tlTrack.getBoundingClientRect();
      const startX = ev.clientX;
      const origT = evt.t_s;
      const wasPlaying = !video.paused;
      if (wasPlaying) video.pause();
      let moved = false;
      m.classList.add("dragging");

      const move = (e2) => {
        const dx = e2.clientX - startX;
        if (!moved && Math.abs(dx) < 2) return;
        moved = true;
        const span2 = state.viewEnd - state.viewStart;
        const dt = (dx / rect.width) * span2;
        const newT = Math.max(0, Math.min(DURATION, origT + dt));
        evt.t_s = +newT.toFixed(3);
        m.style.left = `${((newT - state.viewStart) / span2) * 100}%`;
        m.title = `${evt.event} @ ${fmt(evt.t_s)} — drag to adjust`;
        try { video.currentTime = newT; } catch (_) {}
      };
      const up = () => {
        m.classList.remove("dragging");
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        if (moved) {
          renumberAllSets();
          scheduleSave();
          render();
        } else {
          // Click-without-drag → seek to the event
          try { video.currentTime = evt.t_s; } catch (_) {}
        }
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
  }

  function updateCursor() {
    if (!tlCursor) return;
    const span = state.viewEnd - state.viewStart;
    if (span <= 0) { tlCursor.style.display = "none"; return; }
    const pct = ((video.currentTime - state.viewStart) / span) * 100;
    if (pct < 0 || pct > 100) {
      tlCursor.style.display = "none";
    } else {
      tlCursor.style.display = "";
      tlCursor.style.left = `${pct}%`;
    }
  }

  function updateRangeLabel() {
    if (!tlRange) return;
    const span = state.viewEnd - state.viewStart;
    const isFull = state.viewStart < 0.01 && state.viewEnd > DURATION - 0.01;
    const zoom = DURATION / span;
    tlRange.textContent = isFull
      ? `FULL · ${fmt(0)} → ${fmt(DURATION)}`
      : `${zoom.toFixed(1)}× · ${fmt(state.viewStart)} → ${fmt(state.viewEnd)}`;
  }

  // ── Render ──────────────────────────────────────────────────────────────
  function render() {
    const sorted = [...state.events].sort((a, b) => a.t_s - b.t_s);
    sorted.forEach((e, i) => e.seq = i);

    list.innerHTML = "";
    sorted.forEach((e, idx) => {
      const row = document.createElement("div");
      row.className = "evt";
      row.dataset.eventKind = e.event;
      row.title = "Click to seek · double-click to play · drag to adjust time";
      if ((e.event === "rally_start" || e.event === "rally_end") && e.set != null && e.rally_in_set != null) {
        row.dataset.rallyKey = `${e.set}/${e.rally_in_set}`;
      }
      attachRowDrag(row, e, { timeCell: ".t" });
      row.ondblclick = (ev) => {
        if (ev.target.closest("button")) return;
        try { video.currentTime = e.t_s; video.play().catch(() => {}); } catch (_) {}
      };

      const tCell = document.createElement("span");
      tCell.className = "t";
      tCell.textContent = fmt(e.t_s);
      row.appendChild(tCell);

      const main = document.createElement("span");
      const kindSpan = document.createElement("span");
      kindSpan.className = `kind ${e.event}`;
      kindSpan.textContent = e.event;
      main.appendChild(kindSpan);

      const meta = document.createElement("span");
      meta.className = "meta";
      const parts = [];
      if (e.set != null) parts.push(`set ${e.set}`);
      if (e.rally_in_set != null) parts.push(`r${e.rally_in_set}`);
      meta.textContent = parts.length ? "  " + parts.join(" · ") : "";
      main.appendChild(meta);

      if (e.event === "rally_start" || e.event === "rally_end") {
        const flipBtn = document.createElement("button");
        flipBtn.className = "flip-btn";
        flipBtn.textContent = "⇄";
        flipBtn.title = e.event === "rally_start"
          ? "Convert to rally_end"
          : "Convert to rally_start";
        flipBtn.onclick = (ev) => { ev.stopPropagation(); flipRallyEvent(e.seq); };
        main.appendChild(flipBtn);
      }
      if (e.event === "rally_end") {
        const scoreBtn = document.createElement("button");
        scoreBtn.className = "score-tag" + (e.score_by ? ` ${e.score_by}` : "");
        scoreBtn.textContent = e.score_by === "home" ? state.teamHome : e.score_by === "away" ? state.teamAway : "?";
        scoreBtn.title = "Click to cycle: we → opp → unset";
        scoreBtn.onclick = (ev) => { ev.stopPropagation(); cycleScore(e.seq); };
        main.appendChild(scoreBtn);
      }
      row.appendChild(main);

      const del = document.createElement("button");
      del.className = "del";
      del.title = "Delete event";
      del.textContent = "✕";
      del.onclick = (ev) => { ev.stopPropagation(); deleteEvent(e.seq); };
      row.appendChild(del);

      list.appendChild(row);
    });

    // Status pills
    const cs = currentSet();
    const op = cs ? openRallyInSet(cs) : null;
    const hp = cs ? highestRallyInSet(cs) : 0;
    setPill.textContent  = `SET ${cs ?? "—"}`;
    pointPill.textContent = `RALLY ${hp || "—"}`;
    statePill.textContent = op ? "RALLY OPEN" : "NO RALLY OPEN";
    statePill.classList.toggle("live", !!op);

    // Score pill
    if (scorePill) {
      if (cs) {
        const { home, away } = setScores(cs);
        scorePill.innerHTML = `<span class="home">${home}</span><span class="dash">·</span><span class="away">${away}</span>`;
        scorePill.style.display = "";
      } else {
        scorePill.style.display = "none";
      }
    }

    statusText.textContent = `${sorted.length} EVENTS${state.dirty ? " · SAVING" : ""}`;
    updatePaneStat();

    renderTimeline(sorted);
    renderRalliesView();
    if (document.getElementById("tab-highlights").style.display !== "none") {
      renderHighlightsView();
    }
    updateCursor();
    updateRangeLabel();
    renderScorebug();
    updateScoreCursor();
    _prevNowKey = null;             // force highlight recompute after re-render
    updateNowPlayingHighlight();
  }

  function flash(msg) {
    statusText.textContent = msg;
    setTimeout(render, 1500);
  }

  // ── AI-pending rally confirmation ───────────────────────────────────────
  function confirmAIRally(startEvent, endEvent) {
    startEvent.confirmed = true;
    if (endEvent) endEvent.confirmed = true;
    scheduleSave();
    render();
  }

  // (Auto-confirm-via-playback removed — confirmation is now manual only via the ✓ button.)
  function tickPendingConfirmation() { /* no-op */ }

  // ── Highlight the currently-playing rally in both tabs ─────────────────
  let _prevNowKey = null;
  function updateNowPlayingHighlight() {
    const t = video.currentTime;
    const segs = getRallySegments();
    const inSeg = segs.find(s => t >= s.start && t <= s.end);
    const key = inSeg ? `${inSeg.set}/${inSeg.rally_in_set}` : null;
    if (key === _prevNowKey) return;
    _prevNowKey = key;
    document.querySelectorAll(".now").forEach(el => el.classList.remove("now"));
    if (!key) return;
    document.querySelectorAll(`[data-rally-key="${CSS.escape(key)}"]`).forEach(el => el.classList.add("now"));
  }

  // ── Score badge that follows the playhead on the timeline ───────────────
  function updateScoreCursor() {
    const el = document.getElementById("tl-score-now");
    if (!el) return;
    const t = video.currentTime;
    const cs = currentSetAtTime(t);
    if (!cs) { el.innerHTML = ""; return; }
    const { home, away } = setScoresAtTime(cs, t);
    el.innerHTML =
      `<span class="home">${home}</span>` +
      `<span class="dash">·</span>` +
      `<span class="away">${away}</span>`;
  }

  // ── Rallies view (Hudl-style hierarchical) ──────────────────────────────
  function renderRalliesView() {
    const tab = document.getElementById("tab-rallies");
    if (!tab) return;
    tab.innerHTML = "";

    const sorted = [...state.events].sort((a, b) => a.t_s - b.t_s);

    // Group sets and pair up rallies
    const sets = [];
    let cur = null;
    for (const e of sorted) {
      if (e.event === "set_start") {
        cur = { num: e.set, rallies: [] };
        sets.push(cur);
      } else if (e.event === "set_end") {
        if (cur) cur.ended = true;
      } else if (e.event === "rally_start") {
        if (!cur) { cur = { num: 1, rallies: [] }; sets.push(cur); }
        cur.rallies.push({ num: e.rally_in_set, startEvent: e });
      } else if (e.event === "rally_end") {
        if (cur && cur.rallies.length) {
          const last = cur.rallies[cur.rallies.length - 1];
          if (!last.endEvent) last.endEvent = e;
        }
      }
    }

    if (!sets.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "NO RALLIES YET · PRESS SPACE TO START A RALLY";
      tab.appendChild(empty);
      return;
    }

    const openRally = (() => {
      for (let i = sorted.length - 1; i >= 0; i--) {
        const e = sorted[i];
        if (e.event === "rally_end") return null;
        if (e.event === "rally_start") return e;
      }
      return null;
    })();

    for (const set of sets) {
      const { home, away } = setScores(set.num);
      const scan = state.activeScans[set.num];
      const header = document.createElement("div");
      header.className = "set-header" + (scan ? " scanning" : "");
      const btnLabel = scan ? "STOP" : "AUTO";
      const btnClass = scan ? "set-auto-btn stop" : "set-auto-btn";
      const btnTitle = scan
        ? "Cancel auto-detect (rallies found so far are kept)"
        : "Auto-detect rallies in / beyond this set";
      header.innerHTML =
        `<span class="set-num-wrap">` +
          `<span class="set-num">Set ${set.num}</span>` +
          `<button class="${btnClass}" title="${btnTitle}">${btnLabel}</button>` +
        `</span>` +
        `<span class="set-meta">${set.rallies.length} ` +
          `${set.rallies.length === 1 ? "rally" : "rallies"}</span>` +
        `<span class="set-score">` +
          `<span class="home">${home}</span>` +
          `<span class="dash">·</span>` +
          `<span class="away">${away}</span>` +
        `</span>`;
      tab.appendChild(header);

      const autoBtn = header.querySelector(".set-auto-btn");
      autoBtn.onclick = (ev) => {
        ev.stopPropagation();
        if (state.activeScans[set.num]) {
          if (confirm("Stop the auto-detect for this set? Rallies found so far stay.")) {
            cancelScan(set.num);
          }
        } else {
          showAutoDetectMenu(autoBtn, set.num);
        }
      };

      if (scan) {
        const progressRow = document.createElement("div");
        progressRow.className = "scan-progress";
        progressRow.innerHTML =
          `<div class="scan-bar"><div class="scan-fill" style="width:${scan.pct}%"></div></div>` +
          `<div class="scan-text">` +
            `<span class="scan-status">${escapeText(scan.progress || "Scanning…")}</span>` +
            (scan.lastRallyLabel
              ? ` · <span class="scan-last">latest: ${escapeText(scan.lastRallyLabel)}</span>`
              : "") +
          `</div>`;
        tab.appendChild(progressRow);
      }

      for (const r of set.rallies) {
        const isOpen = !r.endEvent;
        const dur = r.endEvent ? (r.endEvent.t_s - r.startEvent.t_s) : null;
        const scoreBy = r.endEvent?.score_by || null;

        const pending = !!(r.startEvent.auto && r.startEvent.confirmed === false);
        const row = document.createElement("div");
        row.className = "rally-row" + (isOpen ? " open" : "") + (pending ? " pending" : "");
        row.title = pending
          ? "AI-detected · play through to confirm, or use ✓ / ✗"
          : "Click to seek · double-click to play · drag to shift the rally";
        row.dataset.rallyKey = `${set.num}/${r.num}`;
        row.dataset.start = r.startEvent.t_s;
        if (r.endEvent) row.dataset.end = r.endEvent.t_s;
        attachRowDrag(row, r.startEvent, {
          timeCell: ".time",
          linkedEvent: r.endEvent || null,
        });
        row.ondblclick = (ev) => {
          if (ev.target.closest("button, .coll-chip")) return;
          try { video.currentTime = r.startEvent.t_s; video.play().catch(() => {}); } catch (_) {}
        };

        const memberships = collectionsContainingRally(set.num, r.num);

        // Running score at end of this rally (or start if still open)
        const scoreAt = r.endEvent
          ? setScoresAtTime(set.num, r.endEvent.t_s)
          : setScoresAtTime(set.num, r.startEvent.t_s);

        const dropParts = [];
        dropParts.push(`<span class="bullet">${isOpen ? "●" : "▸"}</span>`);
        dropParts.push(`<button class="add-coll-btn" title="Add to highlight collection">+</button>`);
        dropParts.push(`<span class="num">Rally ${r.num}</span>`);
        dropParts.push(`<span class="time">${fmtShort(r.startEvent.t_s)}</span>`);
        dropParts.push(`<span class="dur ${isOpen ? "live" : ""}">${
          isOpen ? "OPEN" : `${Math.round(dur)}s`
        }</span>`);

        // Running-score chip: shows the set score AFTER this rally.
        // Coloring: home/away digit highlighted if that side won this rally.
        if (r.endEvent) {
          const winClass = scoreBy === "home" ? "win-h"
                         : scoreBy === "away" ? "win-a" : "";
          dropParts.push(
            `<span class="score-run ${winClass}" data-seq="${r.endEvent.seq}" title="Click to cycle who scored">` +
              `<span class="h">${scoreAt.home}</span>` +
              `<span class="dash">·</span>` +
              `<span class="a">${scoreAt.away}</span>` +
            `</span>`
          );
        } else {
          dropParts.push(
            `<span class="score-run placeholder">` +
              `<span class="h">${scoreAt.home}</span>` +
              `<span class="dash">·</span>` +
              `<span class="a">${scoreAt.away}</span>` +
            `</span>`
          );
        }
        if (pending) {
          dropParts.push(`<button class="ai-confirm" title="Confirm AI rally">✓</button>`);
        }
        dropParts.push(`<button class="del" data-start-seq="${r.startEvent.seq}" title="${pending ? "Reject this AI rally" : "Delete rally"}">✕</button>`);
        row.innerHTML = dropParts.join("");
        if (pending) {
          // Inject AI badge before the rally num
          const aiBadge = document.createElement("span");
          aiBadge.className = "ai-badge";
          aiBadge.textContent = "AI";
          aiBadge.title = "Pending — play through or click ✓ to confirm";
          const numCell = row.querySelector(".num");
          if (numCell) numCell.parentElement.insertBefore(aiBadge, numCell);
        }

        // (Click-to-seek + drag-to-shift handled by attachRowDrag above)
        const tag = row.querySelector(".score-run[data-seq]");
        if (tag && r.endEvent) {
          tag.onclick = (ev) => { ev.stopPropagation(); cycleScore(r.endEvent.seq); };
        }
        const addBtn = row.querySelector(".add-coll-btn");
        if (addBtn) {
          addBtn.onclick = (ev) => {
            ev.stopPropagation();
            showAddToCollectionMenu(addBtn, set.num, r.num);
          };
        }
        const confirmBtn = row.querySelector(".ai-confirm");
        if (confirmBtn) {
          confirmBtn.onclick = (ev) => {
            ev.stopPropagation();
            confirmAIRally(r.startEvent, r.endEvent);
          };
        }
        // Chips for current memberships
        if (memberships.length > 0) {
          const chipRow = document.createElement("div");
          chipRow.className = "rally-chips";
          memberships.forEach(c => {
            const chip = document.createElement("span");
            chip.className = "coll-chip";
            chip.dataset.collId = c.id;
            chip.innerHTML = `${escapeText(c.name)} <span class="x" title="Remove">×</span>`;
            chip.onclick = (ev) => {
              ev.stopPropagation();
              apiRemoveRally(c.id, set.num, r.num);
            };
            chipRow.appendChild(chip);
          });
          row.appendChild(chipRow);
        }
        const del = row.querySelector(".del");
        if (del) {
          del.onclick = (ev) => {
            ev.stopPropagation();
            // delete both start and end of this rally if present
            state.events = state.events.filter(e =>
              e.seq !== r.startEvent.seq &&
              (!r.endEvent || e.seq !== r.endEvent.seq)
            );
            renumberAllSets();
            scheduleSave();
            render();
          };
        }
        tab.appendChild(row);
      }
    }
  }

  function switchTab(name) {
    state.activeTab = name;
    document.querySelectorAll(".pane-tab").forEach(t => {
      t.classList.toggle("active", t.dataset.tab === name);
    });
    document.getElementById("tab-rallies").style.display    = name === "rallies"    ? "" : "none";
    document.getElementById("tab-highlights").style.display = name === "highlights" ? "" : "none";
    document.getElementById("tab-events").style.display     = name === "events"     ? "" : "none";
    if (name === "highlights") renderHighlightsView();
    updatePaneStat();
  }

  function updatePaneStat() {
    const stat = document.getElementById("pane-stat");
    if (!stat) return;
    const suffix = state.dirty ? " · SAVING" : "";
    if (state.activeTab === "events") {
      stat.textContent = `${state.events.length} EVENTS${suffix}`;
    } else if (state.activeTab === "rallies") {
      const n = state.events.filter(e => e.event === "rally_start").length;
      stat.textContent = `${n} ${n === 1 ? "RALLY" : "RALLIES"}${suffix}`;
    } else if (state.activeTab === "highlights") {
      const active = state.activeCollectionId ? findCollection(state.activeCollectionId) : null;
      if (active) {
        const n = active.rallies.length;
        stat.textContent = `${n} ${n === 1 ? "HIGHLIGHT" : "HIGHLIGHTS"}${suffix}`;
      } else {
        const c = state.highlights.collections.length;
        stat.textContent = `${c} ${c === 1 ? "COLLECTION" : "COLLECTIONS"}${suffix}`;
      }
    }
  }

  // ── Highlights tab ──────────────────────────────────────────────────────
  function computeRallyTimes() {
    const sorted = [...state.events].sort((a, b) => a.t_s - b.t_s);
    const open = new Map();
    const out = new Map();
    for (const e of sorted) {
      const key = `${e.set}/${e.rally_in_set}`;
      if (e.event === "rally_start") open.set(key, e.t_s);
      else if (e.event === "rally_end" && open.has(key)) {
        out.set(key, { start: open.get(key), end: e.t_s });
        open.delete(key);
      }
    }
    return out;
  }

  function showAddToCollectionMenu(btn, setNum, rallyNum) {
    document.querySelectorAll(".coll-menu").forEach(m => m.remove());
    const menu = document.createElement("div");
    menu.className = "coll-menu";
    const colls = state.highlights.collections;
    const membershipIds = new Set(
      collectionsContainingRally(setNum, rallyNum).map(c => c.id)
    );
    if (colls.length === 0) {
      const empty = document.createElement("div");
      empty.className = "coll-menu-empty";
      empty.textContent = "No collections yet";
      menu.appendChild(empty);
    }
    colls.forEach(c => {
      const item = document.createElement("div");
      item.className = "coll-menu-item" + (membershipIds.has(c.id) ? " in" : "");
      item.innerHTML = `<span class="check">${membershipIds.has(c.id) ? "✓" : ""}</span><span class="name">${escapeText(c.name)}</span><span class="count">${c.rallies.length}</span>`;
      item.onclick = () => {
        if (membershipIds.has(c.id)) {
          apiRemoveRally(c.id, setNum, rallyNum);
        } else {
          apiAddRally(c.id, setNum, rallyNum);
        }
        menu.remove();
      };
      menu.appendChild(item);
    });
    const newItem = document.createElement("div");
    newItem.className = "coll-menu-item coll-menu-new";
    newItem.textContent = "+ New collection…";
    newItem.onclick = async () => {
      menu.remove();
      const name = prompt("Collection name (e.g. \"Big Kills\")");
      if (name && name.trim()) {
        const coll = await apiCreateCollection(name.trim());
        if (coll) await apiAddRally(coll.id, setNum, rallyNum);
      }
    };
    menu.appendChild(newItem);

    const rect = btn.getBoundingClientRect();
    menu.style.position = "fixed";
    menu.style.left = `${Math.min(rect.left, window.innerWidth - 240)}px`;
    menu.style.top = `${rect.bottom + 4}px`;
    document.body.appendChild(menu);

    setTimeout(() => {
      document.addEventListener("click", function close(e) {
        if (!menu.contains(e.target)) {
          menu.remove();
          document.removeEventListener("click", close);
        }
      });
    }, 10);
  }

  function renderHighlightsView() {
    const tab = document.getElementById("tab-highlights");
    if (!tab) return;
    tab.innerHTML = "";

    const collections = state.highlights.collections;
    const activeId = state.activeCollectionId;
    const active = activeId ? findCollection(activeId) : null;
    const rallyTimes = computeRallyTimes();

    // Header: collection picker + actions
    const header = document.createElement("div");
    header.className = "hl-coll-bar";
    header.innerHTML = `
      <label class="hl-coll-label">COLLECTION</label>
      <select id="hl-coll-select" class="hl-coll-select">
        ${collections.length === 0
          ? '<option value="">— no collections —</option>'
          : collections.map(c =>
              `<option value="${escapeAttr(c.id)}" ${c.id === activeId ? "selected" : ""}>${escapeText(c.name)} (${c.rallies.length})</option>`
            ).join("")
        }
      </select>
      <button id="hl-new-coll" class="hl-coll-btn" title="Create new collection">+ NEW</button>
      <button id="hl-rename-coll" class="hl-coll-btn" title="Rename" ${!active ? "disabled" : ""}>✎</button>
      <button id="hl-delete-coll" class="hl-coll-btn danger" title="Delete" ${!active ? "disabled" : ""}>✕</button>
    `;
    tab.appendChild(header);

    document.getElementById("hl-coll-select").onchange = (ev) => {
      state.activeCollectionId = ev.target.value || null;
      renderHighlightsView();
    };
    document.getElementById("hl-new-coll").onclick = async () => {
      const name = prompt("Collection name (e.g. \"Big Kills\")");
      if (name && name.trim()) await apiCreateCollection(name.trim());
    };
    if (active) {
      document.getElementById("hl-rename-coll").onclick = async () => {
        const name = prompt("Rename collection:", active.name);
        if (name && name.trim() && name !== active.name) {
          await apiRenameCollection(active.id, name.trim());
        }
      };
      document.getElementById("hl-delete-coll").onclick = async () => {
        if (confirm(`Delete collection "${active.name}"? (Source rallies stay.)`)) {
          await apiDeleteCollection(active.id);
        }
      };
    }

    // Body
    if (!active) {
      const empty = document.createElement("div");
      empty.className = "hl-empty";
      empty.textContent = "CREATE A COLLECTION TO START";
      tab.appendChild(empty);
      return;
    }
    if (active.rallies.length === 0) {
      const empty = document.createElement("div");
      empty.className = "hl-empty";
      empty.textContent = `EMPTY · ADD RALLIES FROM THE RALLIES TAB ( + BUTTON )`;
      tab.appendChild(empty);
    }

    const totalDur = active.rallies.reduce((s, r) => {
      const t = rallyTimes.get(`${r.set}/${r.rally_in_set}`);
      return s + (t ? t.end - t.start : 0);
    }, 0);

    // Each rally is its own row
    for (const r of active.rallies) {
      const key = `${r.set}/${r.rally_in_set}`;
      const times = rallyTimes.get(key);
      const row = document.createElement("div");
      row.className = "hl-row";
      row.dataset.rallyKey = key;
      const dur = times ? `${(times.end - times.start).toFixed(1)}s` : "—";
      const startT = times ? fmt(times.start) : "—";
      row.innerHTML = `
        <span class="num">Set ${r.set} · Rally ${r.rally_in_set}</span>
        <span class="time">${startT}</span>
        <span class="dur">${dur}</span>
        <button class="del" title="Remove from collection">✕</button>
        <textarea placeholder="notes (e.g. \"great cross-court kill\")">${escapeText(r.notes || "")}</textarea>
      `;
      row.onclick = (ev) => {
        if (ev.target.closest("textarea, button")) return;
        if (times) video.currentTime = times.start;
      };
      row.ondblclick = (ev) => {
        if (ev.target.closest("textarea, button")) return;
        if (times) { video.currentTime = times.start; video.play().catch(() => {}); }
      };
      row.querySelector(".del").onclick = (ev) => {
        ev.stopPropagation();
        apiRemoveRally(active.id, r.set, r.rally_in_set);
      };
      const ta = row.querySelector("textarea");
      ta.oninput = () => {
        r.notes = ta.value;
        apiUpdateRallyNotes(active.id, r.set, r.rally_in_set, ta.value);
      };
      tab.appendChild(row);
    }

    // Footer: name + scorebug toggle + export
    if (active.rallies.length > 0) {
      const footer = document.createElement("div");
      footer.className = "hl-actions";
      footer.innerHTML = `
        <input type="text" id="hl-name" placeholder="reel filename" value="${escapeAttr(active.name)}"
               style="flex:2; background: var(--bg-input); color: var(--text);
                      border: 1px solid var(--border-strong); border-radius: 2px;
                      padding: 6px 8px; font: 12px/1 var(--f-mono);"
               autocomplete="off" />
        <label class="hl-sb-toggle">
          <input type="checkbox" id="hl-with-sb" />
          <span>burn scorebug</span>
        </label>
        <button id="hl-export" class="primary">⤓ Export · ${fmt(totalDur)}</button>
      `;
      tab.appendChild(footer);
      document.getElementById("hl-export").onclick = () => {
        const nameInput = document.getElementById("hl-name");
        const withSb = document.getElementById("hl-with-sb").checked;
        exportReel(nameInput.value || active.id, active.id, withSb);
      };
    }
  }

  // ── Set time ranges (for per-set auto-detect) ───────────────────────────
  function setTimeRange(setNum) {
    const sorted = [...state.events].sort((a, b) => a.t_s - b.t_s);
    const setStart = sorted.find(e => e.event === "set_start" && e.set === setNum);
    const setEnd   = sorted.find(e => e.event === "set_end"   && e.set === setNum);
    let start = setStart ? setStart.t_s : null;
    let end   = setEnd ? setEnd.t_s : null;
    if (start === null) {
      const firstRally = sorted.find(e => e.event === "rally_start" && e.set === setNum);
      if (firstRally) start = firstRally.t_s;
    }
    if (end === null) {
      const nextSet = sorted.find(e => e.event === "set_start" && e.set > setNum);
      if (nextSet) end = nextSet.t_s;
      else {
        const lastRally = [...sorted].reverse().find(e => e.event === "rally_end" && e.set === setNum);
        end = lastRally ? Math.min(DURATION, lastRally.t_s + 60) : DURATION;
      }
    }
    if (start === null || end === null || end <= start) return null;
    return { start, end };
  }
  function setBeyondRange(setNum) {
    // From end of this set to start of next set (or end of video)
    const sorted = [...state.events].sort((a, b) => a.t_s - b.t_s);
    const thisSet = setTimeRange(setNum);
    if (!thisSet) return null;
    const start = thisSet.end;
    const nextSet = sorted.find(e => e.event === "set_start" && e.set > setNum);
    const end = nextSet ? nextSet.t_s : DURATION;
    if (end - start < 5) return null;
    return { start, end };
  }

  function showAutoDetectMenu(btn, setNum) {
    document.querySelectorAll(".auto-menu").forEach(m => m.remove());
    const menu = document.createElement("div");
    menu.className = "coll-menu auto-menu";

    const within = setTimeRange(setNum);
    const beyond = setBeyondRange(setNum);

    const mkItem = (label, range, attribSet) => {
      const item = document.createElement("div");
      item.className = "coll-menu-item";
      const minutes = Math.ceil((range.end - range.start) / 60);
      item.innerHTML =
        `<span class="check"></span>` +
        `<span class="name">${label}</span>` +
        `<span class="count">${fmtShort(range.start)}→${fmtShort(range.end)} · ${minutes}m</span>`;
      item.onclick = () => {
        menu.remove();
        autoDetectRallies({
          start_t: range.start,
          end_t:   range.end,
          set_num: attribSet,
          label,
        });
      };
      menu.appendChild(item);
    };
    if (within) mkItem(`Within Set ${setNum}`,  within, setNum);
    if (beyond) mkItem(`Beyond Set ${setNum}`, beyond, setNum + 1);
    if (!within && !beyond) {
      const empty = document.createElement("div");
      empty.className = "coll-menu-empty";
      empty.textContent = "No time range to scan for this set";
      menu.appendChild(empty);
    }

    const rect = btn.getBoundingClientRect();
    menu.style.position = "fixed";
    menu.style.left = `${Math.min(rect.left, window.innerWidth - 260)}px`;
    menu.style.top = `${rect.bottom + 4}px`;
    document.body.appendChild(menu);
    setTimeout(() => {
      document.addEventListener("click", function close(e) {
        if (!menu.contains(e.target)) {
          menu.remove();
          document.removeEventListener("click", close);
        }
      });
    }, 10);
  }

  // ── Streaming auto-detect ───────────────────────────────────────────────
  async function autoDetectRallies(opts = {}) {
    const s = opts.start_t !== undefined ? opts.start_t : state.viewStart;
    const e = opts.end_t   !== undefined ? opts.end_t   : state.viewEnd;
    const setNum = opts.set_num || 0;
    const targetSetForActive = opts.clickedSet || setNum;
    const rangeLabel = opts.label
      || (s < 0.01 && e > DURATION - 0.01 ? "FULL game" : `${fmtShort(s)} → ${fmtShort(e)}`);

    if (state.activeScans[targetSetForActive]) {
      // Already running for this set — ignore (button should be in STOP mode)
      return;
    }

    // Kick off the job
    let started;
    try {
      const r = await fetch("/api/auto-rallies/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start_t: s, end_t: e, set_num: setNum || undefined }),
      });
      started = await r.json();
    } catch (err) {
      alert("Failed to start detection: " + err.message);
      return;
    }
    if (!started.ok) {
      alert("Failed to start detection: " + (started.error || "unknown"));
      return;
    }

    state.activeScans[targetSetForActive] = {
      jobId: started.job_id,
      pct: 0,
      progress: "Starting…",
      ralliesCount: 0,
      lastRallyLabel: "",
      rangeLabel,
    };
    render();
    pollScan(targetSetForActive);
  }

  async function pollScan(setNum) {
    while (true) {
      await new Promise(r => setTimeout(r, 900));
      const scan = state.activeScans[setNum];
      if (!scan) return;   // user cancelled / cleared

      let job;
      try {
        const r = await fetch(`/api/auto-rallies/status/${scan.jobId}`);
        job = await r.json();
      } catch (err) {
        // network blip — try again
        continue;
      }

      // Refresh events to reflect newly-inserted rallies
      try {
        const er = await fetch("/api/events");
        const ed = await er.json();
        state.events = (ed.events || []).map(ev => {
          if (ev.score_by === "we")  ev.score_by = "home";
          if (ev.score_by === "opp") ev.score_by = "away";
          return ev;
        });
      } catch (err) {}

      // Update scan state
      scan.pct = job.pct ?? scan.pct;
      scan.progress = job.progress || scan.progress;
      scan.ralliesCount = job.rallies_count ?? scan.ralliesCount;
      scan.lastRallyLabel = job.last_rally_label || scan.lastRallyLabel;
      render();

      if (job.status === "done" || job.status === "cancelled" || job.status === "error") {
        delete state.activeScans[setNum];
        if (renumberAllSets()) scheduleSave();
        render();
        if (job.status === "done") {
          flash(`✓ Detected ${scan.ralliesCount} rallies`);
        } else if (job.status === "cancelled") {
          flash(`⏹ Cancelled · ${scan.ralliesCount} rallies kept`);
        } else {
          alert("Detection failed: " + (job.error || "unknown"));
        }
        return;
      }
    }
  }

  async function cancelScan(setNum) {
    const scan = state.activeScans[setNum];
    if (!scan) return;
    try {
      await fetch(`/api/auto-rallies/cancel/${scan.jobId}`, { method: "POST" });
    } catch (err) {
      console.error("cancel failed", err);
    }
  }

  async function exportReel(name, collectionId, withScorebug) {
    const tab = document.getElementById("tab-highlights");
    const btn = document.getElementById("hl-export");
    if (btn) {
      btn.disabled = true;
      btn.textContent = withScorebug
        ? "Encoding scorebug overlay… (this takes a couple of minutes)"
        : "Exporting…";
    }

    let result;
    try {
      const r = await fetch("/api/export-reel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name || "",
          collection_id: collectionId || "",
          with_scorebug: !!withScorebug,
          team_home: state.teamHome,
          team_away: state.teamAway,
        }),
      });
      result = await r.json();
    } catch (e) {
      result = { error: "Network error" };
    }

    if (btn) { btn.disabled = false; btn.textContent = "⤓ Export"; }

    const old = tab.querySelector(".hl-result");
    if (old) old.remove();
    const note = document.createElement("div");
    note.className = "hl-result";
    if (result.ok) {
      const dl = `/api/reel-download?name=${encodeURIComponent(name || "")}`;
      note.innerHTML = `✓ Exported ${result.segments} clips · ${result.size_mb} MB · ${result.total_duration_s}s` +
        `<br><span class="muted mono" style="font-size:10px">${escapeText(result.path)}</span>` +
        `<a href="${dl}">download</a>`;
    } else {
      note.style.color = "var(--c-danger)";
      note.textContent = `✗ ${result.error || "Export failed"}`;
    }
    tab.appendChild(note);
  }

  function escapeAttr(s) { return String(s).replace(/[&"<>]/g, c => ({"&":"&amp;",'"':"&quot;","<":"&lt;",">":"&gt;"}[c])); }
  function escapeText(s) { return String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

  // ── Player controls (shared with labeller) ──────────────────────────────
  function bumpRate(dir) {
    const i = RATES.findIndex(r => Math.abs(r - video.playbackRate) < 0.01);
    let j = (i === -1) ? RATES.indexOf(1.0) : i + dir;
    j = Math.max(0, Math.min(RATES.length - 1, j));
    video.playbackRate = RATES[j];
  }
  function resetRate() { video.playbackRate = 1.0; }
  function seekBy(dt) {
    const t = Math.max(0, Math.min(video.duration || DURATION, video.currentTime + dt));
    video.currentTime = t;
  }
  function frameStep(dir) {
    video.pause();
    seekBy(dir * (1 / FPS));
  }

  video.addEventListener("timeupdate", () => {
    if (nowEl) nowEl.textContent = fmt(video.currentTime);
    updateCursor();
    updateNowPlayingHighlight();
    updateScoreCursor();
    renderScorebug();   // keep scorebug synced to playhead
    tickPendingConfirmation();
    if (state.rallyOnly && !video.paused) {
      const t = video.currentTime;
      const segs = getRallySegments();
      if (!segs.length) return;
      const inSeg = segs.find(s => t >= s.start && t < s.end);
      if (!inSeg) {
        const next = segs.find(s => s.start > t);
        if (next) video.currentTime = next.start;
        else video.pause();
      }
    }
  });
  video.addEventListener("play", () => { jumpToRallyIfNeeded(); });
  video.addEventListener("seeked", () => {
    updateCursor();
    updateNowPlayingHighlight();
    updateScoreCursor();
    renderScorebug();
  });
  video.addEventListener("ratechange", () => {
    if (rateSelectEl) {
      const v = video.playbackRate;
      const closest = [...rateSelectEl.options].reduce((best, o) => {
        const d = Math.abs(parseFloat(o.value) - v);
        return d < best.d ? { d, o } : best;
      }, { d: Infinity, o: null });
      if (closest.o && closest.d < 0.01) rateSelectEl.value = closest.o.value;
    }
  });

  // ── UI wiring ───────────────────────────────────────────────────────────
  function wireUI() {
    document.getElementById("btn-game-start").onclick    = () => stamp("game_start");
    document.getElementById("btn-set-start").onclick     = () => stamp("set_start");
    document.getElementById("btn-point-toggle").onclick  = toggleRally;
    document.getElementById("btn-set-end").onclick       = () => stamp("set_end");
    document.getElementById("btn-game-end").onclick      = () => stamp("game_end");
    document.getElementById("btn-rewind").onclick        = () => seekBy(-5);
    document.getElementById("btn-fwd").onclick           = () => seekBy(+5);
    document.getElementById("btn-slower").onclick        = () => bumpRate(-1);
    document.getElementById("btn-faster").onclick        = () => bumpRate(+1);
    document.getElementById("btn-reset-rate").onclick    = resetRate;
    document.getElementById("btn-export").onclick        = exportJson;

    const rallyBtn = document.getElementById("btn-rally-only");
    if (rallyBtn) rallyBtn.onclick = () => setRallyOnly(!state.rallyOnly);

    const homeBtn = document.getElementById("btn-score-home");
    const awayBtn = document.getElementById("btn-score-away");
    if (homeBtn) homeBtn.onclick = () => scoreRally("home");
    if (awayBtn) awayBtn.onclick = () => scoreRally("away");

    const zin  = document.getElementById("btn-zoom-in");
    const zout = document.getElementById("btn-zoom-out");
    const zrst = document.getElementById("btn-zoom-reset");
    if (zin)  zin.onclick  = zoomInAtPlayhead;
    if (zout) zout.onclick = zoomOutAtPlayhead;
    if (zrst) zrst.onclick = zoomReset;

    document.querySelectorAll(".scorebug-picker .sb-pos").forEach(btn => {
      btn.onclick = () => setScorebugPos(btn.dataset.pos);
    });

    document.querySelectorAll(".pane-tab").forEach(t => {
      t.onclick = () => switchTab(t.dataset.tab);
    });

    const teamsBtn = document.getElementById("btn-edit-teams");
    if (teamsBtn) teamsBtn.onclick = editTeams;

    if (rateSelectEl) {
      rateSelectEl.addEventListener("change", () => {
        video.playbackRate = parseFloat(rateSelectEl.value);
      });
    }
    if (tlTrack) {
      // Drag-to-pan, click-to-seek (differentiated by drag distance)
      tlTrack.addEventListener("mousedown", (ev) => {
        if (ev.button !== 0) return;
        const rect = tlTrack.getBoundingClientRect();
        const startX = ev.clientX;
        const startViewStart = state.viewStart;
        const startViewEnd   = state.viewEnd;
        let moved = false;
        const move = (e) => {
          const dx = e.clientX - startX;
          if (!moved && Math.abs(dx) < 4) return;
          moved = true;
          tlTrack.classList.add("dragging");
          const span = startViewEnd - startViewStart;
          const dt = -(dx / rect.width) * span;
          setView(startViewStart + dt, startViewEnd + dt);
        };
        const up = (e) => {
          tlTrack.classList.remove("dragging");
          document.removeEventListener("mousemove", move);
          document.removeEventListener("mouseup", up);
          if (!moved) {
            const ratio = (e.clientX - rect.left) / rect.width;
            const span = state.viewEnd - state.viewStart;
            const t = Math.max(0, Math.min(DURATION, state.viewStart + ratio * span));
            video.currentTime = t;
          }
        };
        document.addEventListener("mousemove", move);
        document.addEventListener("mouseup", up);
      });
      tlTrack.addEventListener("wheel", (e) => {
        e.preventDefault();
        const rect = tlTrack.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const span = state.viewEnd - state.viewStart;
        if (e.shiftKey) {
          const delta = (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY);
          const dx = (delta / 100) * span * 0.4;
          setView(state.viewStart + dx, state.viewEnd + dx);
        } else {
          const factor = e.deltaY > 0 ? 1 / 0.85 : 0.85;
          zoomAt(ratio, factor);
        }
      }, { passive: false });
    }
    document.addEventListener("keydown", onKey);
  }

  async function exportJson() {
    await saveNow();
    const r = await fetch("/api/events");
    const data = await r.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `event_log_${document.body.dataset.gameKey}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    const k = e.key;
    if (k === " ")          { e.preventDefault(); toggleRally(); return; }
    if (k === "g")          { e.preventDefault(); stamp("game_start"); return; }
    if (k === "G")          { e.preventDefault(); stamp("game_end"); return; }
    if (k === "s")          { e.preventDefault(); stamp("set_start"); return; }
    if (k === "S")          { e.preventDefault(); stamp("set_end"); return; }
    if (k === "w" || k === "W") { e.preventDefault(); scoreRally("home"); return; }
    if (k === "o" || k === "O") { e.preventDefault(); scoreRally("away"); return; }
    if (k === "r" || k === "R") { e.preventDefault(); setRallyOnly(!state.rallyOnly); return; }
    if (k === "b" || k === "B") { e.preventDefault(); cycleScorebug(); return; }
    if (k === "Backspace")  { e.preventDefault(); deleteLast(); return; }
    if (k === "u")          { e.preventDefault(); bumpLastTime(-0.5); return; }
    if (k === "U")          { e.preventDefault(); bumpLastTime(+0.5); return; }

    // Zoom
    if (k === "+" || k === "=") { e.preventDefault(); zoomInAtPlayhead(); return; }
    if (k === "-" || k === "_") { e.preventDefault(); zoomOutAtPlayhead(); return; }
    if (k === "0")              { e.preventDefault(); zoomReset(); return; }

    // Player controls
    if (k === "[")  { e.preventDefault(); bumpRate(-1); return; }
    if (k === "]")  { e.preventDefault(); bumpRate(+1); return; }
    if (k === "\\") { e.preventDefault(); resetRate(); return; }
    if (k === "j")  { e.preventDefault(); seekBy(-5);  return; }
    if (k === "l")  { e.preventDefault(); seekBy(+5);  return; }
    if (k === "ArrowLeft")  { e.preventDefault(); seekBy(-5); return; }
    if (k === "ArrowRight") { e.preventDefault(); seekBy(+5); return; }
    if (k === "J")  { e.preventDefault(); seekBy(-10); return; }
    if (k === "L")  { e.preventDefault(); seekBy(+10); return; }
    if (k === "h")  { e.preventDefault(); seekBy(-30); return; }
    if (k === ";")  { e.preventDefault(); seekBy(+30); return; }
    if (k === ",")  { e.preventDefault(); frameStep(-1); return; }
    if (k === ".")  { e.preventDefault(); frameStep(+1); return; }
    if (k === "p" || k === "P") {
      e.preventDefault();
      if (video.paused) video.play(); else video.pause();
      return;
    }
  }

  function fmt(t) {
    t = Math.max(0, t || 0);
    const h = Math.floor(t / 3600);
    const m = Math.floor((t % 3600) / 60);
    const s = Math.floor(t % 60);
    const f = Math.floor((t - Math.floor(t)) * FPS);
    const pad = (n, w=2) => String(n).padStart(w, "0");
    return `${pad(h)}:${pad(m)}:${pad(s)}:${pad(f)}`;
  }

  // Short M:SS or H:MM:SS — used in the side panel where space is tight
  function fmtShort(t) {
    t = Math.max(0, t || 0);
    const h = Math.floor(t / 3600);
    const m = Math.floor((t % 3600) / 60);
    const s = Math.floor(t % 60);
    const pad = (n) => String(n).padStart(2, "0");
    return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
  }

  init();
})();
