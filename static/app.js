/* Breeze Bets desk: vanilla JS over the local API (web_server.py). */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { status: null, slate: null, tracker: null, busy: false, focus: 0, polling: null,
  lastAction: null, editing: null, draft: "", cards: [] };
const SOURCE_LABEL = { ollama: "Written by Ollama", template: "Desk template", edited: "Edited by you" };
const TIER = {
  FREE: { name: "Free play", card: "free card", cls: "tier-free", other: "PREMIUM" },
  PREMIUM: { name: "Premium", card: "premium card", cls: "tier-premium", other: "FREE" },
};
const tierChip = (t) => (TIER[t] ? `<span class="tier ${TIER[t].cls}">${TIER[t].name}</span>` : "");

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const odds = (v) => (v == null ? "n/a" : (v > 0 ? `+${v}` : `${v}`));
const pct = (p, d = 1) => (p == null ? "n/a" : `${(p * 100).toFixed(d)}%`);
const units = (u) => `${Number(u || 0).toFixed(2)}u`;
const signed = (n, d = 2, suffix = "") => (n == null ? "--"
  : `${n > 0 ? "+" : n < 0 ? "-" : ""}${Math.abs(n).toFixed(d)}${suffix}`);
const tone = (n) => (n == null || n === 0 ? "" : n > 0 ? "pos" : "neg");

function tip(iso) {
  if (!iso) return "Time TBD";
  const d = new Date(iso);
  const mins = Math.round((d - Date.now()) / 60000);
  const clock = d.toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
  if (mins <= 0) return `${clock}, started`;
  if (mins < 90) return `${clock}, in ${mins} min`;
  return clock;
}

async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: {} };
  if (init.method !== "GET") {
    init.headers["X-Breeze-Desk"] = "1";
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body || {});
  }
  const res = await fetch(path, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

let toastTimer = null;
function toast(msg, bad = false, undoable = false) {
  const t = $("toast");
  $("toast-msg").textContent = msg;
  $("toast-undo").hidden = !undoable;
  t.classList.toggle("bad", bad);
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, undoable ? 8000 : bad ? 7000 : 3500);
}

// ------------------------------------------------------------------ scoreboard
function renderBoard() {
  const s = state.tracker?.summary;
  const net = $("k-net");
  net.textContent = s ? signed(s.net_units, 2, "u") : "--";
  net.className = s ? tone(s.net_units) : "";
  const roi = $("k-roi");
  roi.textContent = s && s.roi_percent != null ? signed(s.roi_percent, 1, "%") : "--";
  roi.className = s ? tone(s.roi_percent) : "";
  $("k-record").textContent = s ? s.record : "--";
  $("k-winrate").textContent = s && s.win_rate != null ? `${s.win_rate.toFixed(1)}%` : "--";
  $("k-winrate-n").textContent = s ? `${s.picks} settled pick${s.picks === 1 ? "" : "s"}` : "";
  const clv = $("k-clv");
  clv.textContent = s && s.clv_avg != null ? signed(s.clv_avg, 2, "%") : "--";
  clv.className = s ? tone(s.clv_avg) : "";
  $("k-clv-n").textContent = !s ? "" : s.clv_n
    ? `${Math.round((s.clv_beat_rate * s.clv_n) / 100)} of ${s.clv_n} beat the close` : "No closing lines yet";
  $("k-open").textContent = s ? `${s.open_count} / ${units(s.open_units)}` : "--";
  const mode = state.status?.mode;
  document.querySelectorAll(".mode-btn").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
}

function tickClock() {
  const st = state.status;
  if (!st) return;
  const left = Math.max(0, Math.round((new Date(st.next_run) - Date.now()) / 1000));
  const h = String(Math.floor(left / 3600)).padStart(2, "0");
  const m = String(Math.floor((left % 3600) / 60)).padStart(2, "0");
  const s = String(left % 60).padStart(2, "0");
  $("k-next").textContent = `${h}:${m}:${s}`;
}

function renderTicker() {
  const el = $("ticker");
  const jobs = state.status?.jobs || {};
  const verb = { scan: "9:00 scan", settle: "settlement", daily: "daily 9:00 run" };
  el.className = "ticker";
  if (jobs.current) {
    el.classList.add("run");
    el.textContent = `Running ${verb[jobs.current.kind]} (${jobs.current.mode}) since `
      + `${new Date(jobs.current.started_at).toLocaleTimeString()}. Picks refresh when it finishes.`;
  } else if (jobs.last) {
    const j = jobs.last;
    const tail = (j.steps.at(-1)?.tail || "").split("\n").filter(Boolean).at(-1) || "";
    if (j.status !== "ok") el.classList.add("fail");
    el.textContent = `Last ${verb[j.kind]} (${j.mode}, ${j.trigger}) ${j.status === "ok" ? "finished" : "failed"} `
      + `${new Date(j.finished_at).toLocaleTimeString()}. ${tail}`;
  } else {
    const sched = state.status?.scheduler ? "The 9:00 AM run is scheduled." : "Scheduler is off.";
    el.textContent = `No pipeline runs yet this session. ${sched}`;
  }
  const running = Boolean(jobs.current);
  ["act-scan", "act-settle"].forEach((id) => { $(id).disabled = running || state.busy; });
}

// ------------------------------------------------------------------ slate
function ticket(v) {
  const prop = v.kind === "prop";
  const injury = /injur|questionable|doubtful|gtd/i.test(v.hitl_reason);
  const stats = [
    ["Odds", `${odds(v.offered_american)} ${esc(v.bookmaker || "")}`],
    [prop ? "Model" : "Fair", pct(prop ? v.model_prob : v.fair_prob)],
    ["Edge", `+${Number(v.ev_percent || 0).toFixed(2)}% EV`],
  ];
  if (prop && v.projection != null) {
    stats.push(["Projection", `${Number(v.projection).toFixed(1)} (${signed(v.delta, 1)})`]);
  }
  const why = v.hitl_reason
    ? `<p class="t-why">${injury ? '<span class="flag">[Injury] </span>' : ""}${esc(v.hitl_reason)}</p>` : "";
  const move = v.tier && !v.tier_locked && TIER[v.tier]
    ? `<button type="button" class="btn btn-undo" data-act="tier" data-tier="${TIER[v.tier].other}">Move to ${TIER[TIER[v.tier].other].card}</button>`
    : "";
  const act = v.signed
    ? `<div class="signed-row"><span class="stamp">Signed &middot; Captain HZ</span>${tierChip(v.tier)}
         ${move}
         ${v.undoable ? '<button type="button" class="btn btn-undo" data-act="undo">Undo</button>' : ""}</div>
       ${v.post ? postBlock(v) : ""}`
    : `<div class="t-act">
         <button type="button" class="btn btn-approve" data-act="approve" data-tier="FREE">Approve free ${units(v.units)}</button>
         <button type="button" class="btn btn-premium" data-act="approve" data-tier="PREMIUM">Approve premium</button>
         <button type="button" class="btn btn-decline" data-act="decline">Decline</button>
       </div>`;
  return `<article class="ticket${v.signed ? " signed" : ""}" tabindex="0" data-id="${esc(v.pick_id)}"
            aria-label="${esc(v.label)}, ${v.signed ? `signed off, ${TIER[v.tier]?.card || ""}` : "awaiting sign-off"}">
    <div class="stub">
      <span class="stub-units">${units(v.units)}</span>
      <span class="stub-id">${esc(v.pick_id)}</span>
    </div>
    <div class="tbody">
      <div class="t-top"><span>${esc(v.market_label)}</span><span>${esc(tip(v.commence_time))}</span></div>
      <p class="t-label">${esc(v.label)}</p>
      <div class="t-event">${esc(v.event)}</div>
      <dl class="t-stats">${stats.map(([k, val]) => `<div><dt>${k}</dt><dd>${val}</dd></div>`).join("")}</dl>
      ${why}
      ${act}
    </div>
  </article>`;
}

function postBlock(v) {
  const editing = state.editing === v.pick_id;
  const header = v.post.split("\n\n")[0];
  const body = editing
    ? `<pre class="post-text">${esc(header)}</pre>
       <label class="sr-only" for="edit-${esc(v.pick_id)}">Write-up for ${esc(v.label)}</label>
       <textarea class="post-edit" id="edit-${esc(v.pick_id)}" rows="5" maxlength="2000">${esc(state.draft)}</textarea>`
    : `<pre class="post-text">${esc(v.post)}</pre>`;
  const buttons = editing
    ? `<button type="button" class="btn btn-mini btn-save" data-post="save">Save</button>
       <button type="button" class="btn btn-mini" data-post="cancel">Cancel</button>`
    : `<button type="button" class="btn btn-mini" data-post="edit">Edit</button>
       <button type="button" class="btn btn-mini" data-post="regen">Regenerate</button>
       <button type="button" class="btn btn-mini btn-copy" data-post="copy">Copy post</button>`;
  return `<div class="post">
    <div class="post-head"><span>Discord post</span><span class="post-src">${SOURCE_LABEL[v.writeup_source] || ""}</span></div>
    ${body}
    <div class="post-act">${buttons}</div>
  </div>`;
}

function findPick(id) {
  for (const sp of state.slate?.sports || []) {
    const hit = [...sp.games, ...sp.props].find((v) => v.pick_id === id);
    if (hit) return hit;
  }
  return null;
}

async function postAction(pickId, what, btn) {
  const v = findPick(pickId);
  if (!v) return;
  if (what === "copy") {
    const ok = await copyText(v.post);
    toast(ok ? `Copied the ${v.label} post.` : "Copy blocked by the browser. Select the post text instead.", !ok);
    return;
  }
  if (what === "edit") {
    state.editing = pickId; state.draft = v.writeup || "";
    renderSlate();
    document.getElementById(`edit-${pickId}`)?.focus();
    return;
  }
  if (what === "cancel") { state.editing = null; renderSlate(); return; }
  const body = what === "regen" ? { regenerate: true } : { text: state.draft };
  btn.disabled = true;
  if (what === "regen") btn.textContent = "Writing...";
  try {
    await api(`/api/picks/${encodeURIComponent(pickId)}/writeup`, { method: "POST", body });
    if (what === "save") state.editing = null;
    toast(what === "save" ? `Saved your write-up for ${v.label}.` : `New write-up for ${v.label}.`);
    await refreshAll();
  } catch (e) {
    toast(`Write-up not ${what === "save" ? "saved" : "regenerated"}: ${e.message}`, true);
    btn.disabled = false;
  }
}

function pillar(title, picks, solo) {
  return `<div class="pillar${solo ? " solo" : ""}"><h3>${title} <span class="num">(${picks.length})</span></h3>
    ${picks.map(ticket).join("")}</div>`;
}

function renderDeclined(list) {
  const box = $("declined");
  box.hidden = !list.length;
  $("declined-n").textContent = list.length;
  $("declined-list").innerHTML = list.map((v) => histItem(v, "DECLINED")).join("");
}

function renderSlate() {
  const sl = state.slate;
  const box = $("sports");
  if (!sl) return;
  const tc = sl.tier_counts || {};
  const cards = sl.signed_count ? ` (free ${tc.FREE || 0}, premium ${tc.PREMIUM || 0})` : "";
  $("slate-note").textContent = sl.queue_count
    ? `Captain HZ sign-off queue: ${sl.queue_count} waiting, ${sl.signed_count} signed${cards}`
    : `Captain HZ sign-off queue is clear${sl.signed_count ? `, ${sl.signed_count} signed${cards}` : ""}`;
  const stale = $("stale");
  stale.hidden = !sl.stale_open;
  stale.textContent = sl.stale_open
    ? `${sl.stale_open} open pick${sl.stale_open === 1 ? " is" : "s are"} from earlier slates and waiting on final scores. Settle final scores to grade them.`
    : "";
  renderDeclined(sl.declined || []);
  if (!sl.sports.length) {
    box.innerHTML = `<div class="empty"><h2>No actionable picks today</h2>
      <p>Nothing on the ${sl.origin === "MOCK" ? "mock" : "live"} slate cleared the model gate yet.
      Run the 9:00 scan now, or let the desk run it at 9:00 AM.</p></div>`;
    return;
  }
  box.innerHTML = sl.sports.map((sp) => {
    // Show each pillar only when it has picks; one pillar spans the full width.
    const cols = [];
    if (sp.games.length) cols.push(["Game lines", sp.games]);
    if (sp.props.length) cols.push(["Player props", sp.props]);
    return `<section class="sport" aria-label="${sp.league}">
      <div class="sport-head"><span class="sport-name">${sp.league}</span>
        <span class="sport-meta">${sp.queue} to sign, ${sp.signed} signed (${units(sp.signed_units)})</span></div>
      <div class="pillars">${cols.map(([t, p]) => pillar(t, p, cols.length === 1)).join("")}</div>
    </section>`;
  }).join("");
  focusTicket(Math.min(state.focus, tickets().length - 1), false);
}

const tickets = () => Array.from(document.querySelectorAll(".ticket"));
function focusTicket(i, move = true) {
  const all = tickets();
  if (!all.length) return;
  state.focus = Math.max(0, Math.min(i, all.length - 1));
  all.forEach((t, k) => t.classList.toggle("focus", k === state.focus));
  if (move) all[state.focus].focus();
}

// ------------------------------------------------------------------ tracker
function splitRow(name, a, cls = "") {
  return `<tr class="${cls}"><th scope="row">${name}</th><td>${a.record}</td>
    <td>${a.win_rate == null ? "--" : a.win_rate.toFixed(1)}</td><td>${a.risked_units.toFixed(2)}</td>
    <td class="${tone(a.net_units)}">${signed(a.net_units)}</td>
    <td class="${tone(a.roi_percent)}">${a.roi_percent == null ? "--" : signed(a.roi_percent, 1, "%")}</td>
    <td class="${tone(a.clv_avg)}">${a.clv_avg == null ? "--" : signed(a.clv_avg, 2, "%")}</td></tr>`;
}
const groupRow = (label) => `<tr class="grp"><th scope="rowgroup" colspan="7">${label}</th></tr>`;

function renderCurve(points) {
  const el = $("curve");
  if (!points.length) {
    el.innerHTML = `<div class="empty-c">No settled official plays yet. The curve starts at the first graded pick.</div>`;
    return;
  }
  const ys = [0, ...points.map((p) => p.net)];
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const span = hi - lo || 1;
  const w = 100, h = 40;
  const n = ys.length - 1 || 1;
  const xy = ys.map((y, i) => `${(i / n) * w},${h - ((y - lo) / span) * h}`).join(" ");
  const zeroY = h - ((0 - lo) / span) * h;
  el.innerHTML = `<svg viewBox="0 -2 ${w} ${h + 4}" preserveAspectRatio="none" role="img"
      aria-label="Cumulative units ${signed(ys.at(-1), 2, "u")} over ${points.length} settled plays">
    <line class="zero" x1="0" x2="${w}" y1="${zeroY}" y2="${zeroY}" vector-effect="non-scaling-stroke"/>
    <polyline class="line" points="${xy}"/></svg>`;
}

function badge(result) {
  const map = { WIN: ["WIN", "b-win"], LOSS: ["LOSS", "b-loss"], PUSH: ["PUSH", "b-void"],
    CANCELLED: ["VOID", "b-void"], UNSETTLED: ["OPEN", "b-open"], DECLINED: ["PASS", "b-declined"] };
  const [txt, cls] = map[result] || [result, "b-open"];
  return `<span class="badge ${cls}">${txt}</span>`;
}

function histItem(v, badgeKey = v.result) {
  const when = v.settled_at || v.commence_time;
  const undo = v.undoable
    ? `<button type="button" class="btn btn-undo" data-act="undo" data-id="${esc(v.pick_id)}"
         aria-label="Undo ${esc(v.label)}">Undo</button>` : "";
  const net = (v.result === "UNSETTLED" ? `<span>${units(v.units)}</span>`
    : `<span class="${tone(v.net_units)}">${signed(v.net_units, 2, "u")}</span>`) + undo;
  return `<li>${badge(badgeKey)}
    <div><div class="h-label">${esc(v.label)}</div>
      <div class="h-meta">${tierChip(v.tier)} ${v.league} ${odds(v.offered_american)} ${units(v.units)} ${when ? new Date(when).toLocaleDateString([], { month: "short", day: "numeric" }) : ""}${v.clv_percent != null ? ` <span class="${tone(v.clv_percent)}">CLV ${signed(v.clv_percent, 2, "%")}</span>` : ""}</div></div>
    <div class="h-net">${net}</div></li>`;
}

function renderTracker() {
  const tr = state.tracker;
  if (!tr) return;
  $("split-body").innerHTML = splitRow("All official", tr.summary, "all")
    + groupRow("Card") + splitRow("Free plays", tr.by_tier.FREE) + splitRow("Premium", tr.by_tier.PREMIUM)
    + groupRow("Sport") + splitRow("NBA", tr.by_league.NBA) + splitRow("NFL", tr.by_league.NFL)
    + groupRow("Market") + splitRow("Game lines", tr.by_kind.game) + splitRow("Player props", tr.by_kind.prop);
  renderCurve(tr.curve);
  $("open-list").innerHTML = tr.open.length ? tr.open.map((v) => histItem(v)).join("")
    : `<li><span class="none">No open official plays. Approved picks land here.</span></li>`;
  $("settled-list").innerHTML = tr.settled.length ? tr.settled.slice(0, 60).map((v) => histItem(v)).join("")
    : `<li><span class="none">Nothing graded yet. Settle final scores after games end.</span></li>`;
}

async function loadDrop() {
  const d = await api("/api/discord-drop");
  state.cards = d.cards;
  $("cards").innerHTML = d.cards.map((c) => `<section class="card-box" aria-label="${esc(c.label)} card">
      <div class="card-head">${tierChip(c.tier)}
        <span class="card-path">${esc(c.path.split(/[\\/]/).slice(-2).join("/"))}</span>
        <button type="button" class="btn btn-mini btn-copy" data-card="${c.tier}">Copy</button></div>
      <pre class="drop">${esc(c.markdown)}</pre></section>`).join("");
  return d;
}

async function copyCard(tier) {
  try {
    const d = await loadDrop();
    const card = d.cards.find((c) => c.tier === tier);
    const ok = await copyText(card.markdown);
    toast(ok ? `${TIER[tier].name} card copied. Signed-off picks only.`
      : "Copy blocked by the browser. Select the card text in the ledger.", !ok);
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ loading
async function refreshAll() {
  const [status, slate, tracker] = await Promise.all([
    api("/api/status"), api("/api/slate"), api("/api/tracker")]);
  Object.assign(state, { status, slate, tracker });
  renderBoard(); tickClock(); renderTicker(); renderSlate(); renderTracker();
  loadDrop().catch(() => {});
}

async function pollStatus() {
  const wasRunning = Boolean(state.status?.jobs?.current);
  state.status = await api("/api/status");
  renderTicker(); tickClock();
  const running = Boolean(state.status.jobs.current);
  if (wasRunning && !running) {
    const j = state.status.jobs.last;
    toast(j.status === "ok" ? `${j.kind === "settle" ? "Settlement" : "Scan"} finished. Desk refreshed.`
      : `${j.kind === "settle" ? "Settlement" : "Scan"} failed. See the ticker for the last line.`, j.status !== "ok");
    await refreshAll();
  }
  clearTimeout(state.polling);
  state.polling = setTimeout(() => pollStatus().catch(() => {}), running ? 2000 : 15000);
}

// ------------------------------------------------------------------ actions
async function act(pickId, action, tier) {
  if (state.busy) return;
  state.busy = true;
  try {
    const { pick } = await api(`/api/picks/${encodeURIComponent(pickId)}/${action}`,
      { method: "POST", body: tier ? { tier } : {} });
    if (action === "undo") {
      if (state.lastAction?.pickId === pickId) state.lastAction = null;
      toast(`Back in the queue: ${pick.label}`);
    } else if (action === "tier") {
      toast(`${pick.label} moved to the ${TIER[pick.tier].card}.`);
    } else if (action === "approve") {
      state.lastAction = { pickId, label: pick.label };
      toast(`Approved ${pick.label}, ${units(pick.units)}, ${TIER[pick.tier].card}`, false, true);
    } else {
      state.lastAction = { pickId, label: pick.label };
      toast(`Declined ${pick.label}`, false, true);
    }
    await refreshAll();
  } catch (e) {
    const verb = { approve: "approved", decline: "declined", undo: "undone", tier: "moved" }[action];
    toast(`Not ${verb}: ${e.message}`, true);
  } finally {
    state.busy = false;
  }
}

function undoLast() {
  const last = state.lastAction;
  if (!last) { toast("Nothing to undo yet."); return; }
  $("toast").hidden = true;
  act(last.pickId, "undo");
}

async function startJob(kind) {
  try {
    await api(`/api/${kind}`, { method: "POST" });
    toast(kind === "scan" ? "9:00 scan started. Picks refresh when it finishes." : "Settlement started.");
    await pollStatus();
  } catch (e) {
    toast(e.message, true);
  }
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  }
}

function wire() {
  $("sports").addEventListener("click", (e) => {
    const pb = e.target.closest("button[data-post]");
    if (pb) { postAction(pb.closest(".ticket").dataset.id, pb.dataset.post, pb); return; }
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const card = btn.closest(".ticket");
    state.focus = tickets().indexOf(card);
    act(card.dataset.id, btn.dataset.act, btn.dataset.tier);
  });
  $("sports").addEventListener("input", (e) => {
    if (e.target.classList.contains("post-edit")) state.draft = e.target.value;
  });
  $("sports").addEventListener("keydown", (e) => {
    if (!e.target.classList.contains("post-edit")) return;
    const card = e.target.closest(".ticket");
    if (e.key === "Escape") { e.preventDefault(); postAction(card.dataset.id, "cancel"); }
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      postAction(card.dataset.id, "save", card.querySelector("[data-post=save]"));
    }
  });
  $("sports").addEventListener("focusin", (e) => {
    const card = e.target.closest(".ticket");
    if (card) focusTicket(tickets().indexOf(card), false);
  });
  for (const id of ["open-list", "declined-list"]) {
    $(id).addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-act='undo']");
      if (btn) act(btn.dataset.id, "undo");
    });
  }
  $("toast-undo").addEventListener("click", undoLast);
  $("act-scan").addEventListener("click", () => startJob("scan"));
  $("act-settle").addEventListener("click", () => startJob("settle"));
  $("act-approve-all").addEventListener("click", async () => {
    if (state.busy) return;
    state.busy = true;
    try {
      const r = await api("/api/picks/approve-all", { method: "POST" });
      toast(`Signed off ${r.approved.length} qualified pick${r.approved.length === 1 ? "" : "s"}`
        + (r.skipped.length ? `; ${r.skipped.length} held back (${r.skipped[0].reason})` : "."));
      await refreshAll();
    } catch (e) { toast(e.message, true); } finally { state.busy = false; }
  });
  $("act-copy-free").addEventListener("click", () => copyCard("FREE"));
  $("act-copy-premium").addEventListener("click", () => copyCard("PREMIUM"));
  $("cards").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-card]");
    if (btn) copyCard(btn.dataset.card);
  });
  document.querySelectorAll(".mode-btn").forEach((b) => b.addEventListener("click", async () => {
    try {
      state.status = await api("/api/settings", { method: "POST", body: { mode: b.dataset.mode } });
      toast(`Desk switched to ${b.dataset.mode} data.`);
      await refreshAll();
    } catch (e) { toast(e.message, true); }
  }));
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    if (k === "z") { e.preventDefault(); undoLast(); return; }
    const all = tickets();
    if (!all.length) return;
    if (k === "j" || k === "arrowdown") { e.preventDefault(); focusTicket(state.focus + 1); }
    else if (k === "k" || k === "arrowup") { e.preventDefault(); focusTicket(state.focus - 1); }
    else if (k === "a" || k === "p" || k === "d") {
      const card = all[state.focus];
      if (!card || card.classList.contains("signed")) return;
      if (k === "d") act(card.dataset.id, "decline");
      else act(card.dataset.id, "approve", k === "p" ? "PREMIUM" : "FREE");
    } else if (k === "t") {
      const card = all[state.focus];
      const v = card && findPick(card.dataset.id);
      if (!v?.tier) return;
      if (v.tier_locked) { toast(`${v.label} has started; its card is locked.`, true); return; }
      act(v.pick_id, "tier", TIER[v.tier].other);
    }
  });
}

wire();
refreshAll().then(() => pollStatus()).catch((e) => {
  $("ticker").textContent = `Desk API unreachable: ${e.message}. Is web_server.py running?`;
  $("ticker").className = "ticker fail";
});
setInterval(tickClock, 1000);
