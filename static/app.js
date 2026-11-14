/* Breeze Bets desk: vanilla JS over the local API (web_server.py). */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { status: null, slate: null, tracker: null, busy: false, focus: 0, polling: null };

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
function toast(msg, bad = false) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("bad", bad);
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, bad ? 7000 : 3500);
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
  const act = v.signed
    ? `<span class="stamp">Signed &middot; Captain HZ</span>`
    : `<div class="t-act">
         <button type="button" class="btn btn-approve" data-act="approve">Approve ${units(v.units)}</button>
         <button type="button" class="btn btn-decline" data-act="decline">Decline</button>
       </div>`;
  return `<article class="ticket${v.signed ? " signed" : ""}" tabindex="0" data-id="${esc(v.pick_id)}"
            aria-label="${esc(v.label)}, ${v.signed ? "signed off" : "awaiting sign-off"}">
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

function pillar(title, picks, solo) {
  return `<div class="pillar${solo ? " solo" : ""}"><h3>${title} <span class="num">(${picks.length})</span></h3>
    ${picks.map(ticket).join("")}</div>`;
}

function renderSlate() {
  const sl = state.slate;
  const box = $("sports");
  if (!sl) return;
  $("slate-note").textContent = sl.queue_count
    ? `Captain HZ sign-off queue: ${sl.queue_count} waiting, ${sl.signed_count} signed`
    : `Captain HZ sign-off queue is clear${sl.signed_count ? `, ${sl.signed_count} signed` : ""}`;
  const stale = $("stale");
  stale.hidden = !sl.stale_open;
  stale.textContent = sl.stale_open
    ? `${sl.stale_open} open pick${sl.stale_open === 1 ? " is" : "s are"} from earlier slates and waiting on final scores. Settle final scores to grade them.`
    : "";
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
  return `<tr class="${cls}"><td>${name}</td><td>${a.record}</td>
    <td>${a.win_rate == null ? "--" : a.win_rate.toFixed(1)}</td><td>${a.risked_units.toFixed(2)}</td>
    <td class="${tone(a.net_units)}">${signed(a.net_units)}</td>
    <td class="${tone(a.roi_percent)}">${a.roi_percent == null ? "--" : signed(a.roi_percent, 1, "%")}</td></tr>`;
}

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
    CANCELLED: ["VOID", "b-void"], UNSETTLED: ["OPEN", "b-open"] };
  const [txt, cls] = map[result] || [result, "b-open"];
  return `<span class="badge ${cls}">${txt}</span>`;
}

function histItem(v) {
  const when = v.settled_at || v.commence_time;
  const net = v.result === "UNSETTLED" ? units(v.units)
    : `<span class="${tone(v.net_units)}">${signed(v.net_units, 2, "u")}</span>`;
  return `<li>${badge(v.result)}
    <div><div class="h-label">${esc(v.label)}</div>
      <div class="h-meta">${v.league} ${odds(v.offered_american)} ${units(v.units)} ${when ? new Date(when).toLocaleDateString([], { month: "short", day: "numeric" }) : ""}</div></div>
    <div class="h-net">${net}</div></li>`;
}

function renderTracker() {
  const tr = state.tracker;
  if (!tr) return;
  $("split-body").innerHTML = splitRow("All official", tr.summary, "all")
    + splitRow("NBA", tr.by_league.NBA) + splitRow("NFL", tr.by_league.NFL)
    + splitRow("Game lines", tr.by_kind.game) + splitRow("Player props", tr.by_kind.prop);
  renderCurve(tr.curve);
  $("open-list").innerHTML = tr.open.length ? tr.open.map(histItem).join("")
    : `<li><span class="none">No open official plays. Approved picks land here.</span></li>`;
  $("settled-list").innerHTML = tr.settled.length ? tr.settled.slice(0, 60).map(histItem).join("")
    : `<li><span class="none">Nothing graded yet. Settle final scores after games end.</span></li>`;
}

async function loadDrop() {
  const d = await api("/api/discord-drop");
  $("drop").textContent = d.markdown;
  $("drop-path").textContent = `Signed-off picks only. Written to ${d.path.split(/[\\/]/).slice(-3).join("/")}`;
  return d;
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
async function act(pickId, action) {
  if (state.busy) return;
  state.busy = true;
  try {
    const { pick } = await api(`/api/picks/${encodeURIComponent(pickId)}/${action}`, { method: "POST" });
    toast(action === "approve" ? `Signed off: ${pick.label}, ${units(pick.units)}` : `Declined: ${pick.label}`);
    await refreshAll();
  } catch (e) {
    toast(`Not ${action === "approve" ? "approved" : "declined"}: ${e.message}`, true);
  } finally {
    state.busy = false;
  }
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
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const card = btn.closest(".ticket");
    state.focus = tickets().indexOf(card);
    act(card.dataset.id, btn.dataset.act);
  });
  $("sports").addEventListener("focusin", (e) => {
    const card = e.target.closest(".ticket");
    if (card) focusTicket(tickets().indexOf(card), false);
  });
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
  $("act-copy").addEventListener("click", async () => {
    try {
      const d = await loadDrop();
      const ok = await copyText(d.markdown);
      toast(ok ? "Discord drop copied. Signed-off picks only." : "Copy blocked by the browser. Select the drop text below.", !ok);
    } catch (e) { toast(e.message, true); }
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
    const all = tickets();
    if (!all.length) return;
    if (k === "j" || k === "arrowdown") { e.preventDefault(); focusTicket(state.focus + 1); }
    else if (k === "k" || k === "arrowup") { e.preventDefault(); focusTicket(state.focus - 1); }
    else if (k === "a" || k === "d") {
      const card = all[state.focus];
      if (card && !card.classList.contains("signed")) act(card.dataset.id, k === "a" ? "approve" : "decline");
    }
  });
}

wire();
refreshAll().then(() => pollStatus()).catch((e) => {
  $("ticker").textContent = `Desk API unreachable: ${e.message}. Is web_server.py running?`;
  $("ticker").className = "ticker fail";
});
setInterval(tickClock, 1000);
