// static/app.js
const $ = (id) => document.getElementById(id);

function toast(msg, ok=true) {
  const t = $("toast");
  t.textContent = msg;
  t.style.borderColor = ok ? "rgba(30,230,168,.35)" : "rgba(255,77,109,.40)";
  t.classList.add("show");
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => t.classList.remove("show"), 2200);
}

function setPill(state) {
  const dot = $("dot");
  const txt = $("statusText");
  const pill = $("statusPill");

  if (state === "ok") {
    dot.style.background = "rgba(30,230,168,.9)";
    dot.style.boxShadow = "0 0 0 6px rgba(30,230,168,.12)";
    txt.textContent = "Live";
    pill.style.borderColor = "rgba(30,230,168,.25)";
  } else if (state === "warn") {
    dot.style.background = "rgba(255,209,102,.95)";
    dot.style.boxShadow = "0 0 0 6px rgba(255,209,102,.12)";
    txt.textContent = "Warning";
    pill.style.borderColor = "rgba(255,209,102,.25)";
  } else if (state === "bad") {
    dot.style.background = "rgba(255,77,109,.95)";
    dot.style.boxShadow = "0 0 0 6px rgba(255,77,109,.12)";
    txt.textContent = "Fall Detected";
    pill.style.borderColor = "rgba(255,77,109,.25)";
  } else {
    dot.style.background = "rgba(255,255,255,.35)";
    dot.style.boxShadow = "0 0 0 6px rgba(255,255,255,.06)";
    txt.textContent = "Connecting…";
    pill.style.borderColor = "rgba(255,255,255,.12)";
  }
}

async function loadSettings() {
  const r = await fetch("/settings", { cache: "no-store" });
  const j = await r.json();

  $("alertTo").value = j.alert_to || "";
  $("smtpHost").textContent = j.smtp_host || "—";
  $("smtpUser").textContent = j.smtp_user || "—";
  $("smtpPort").textContent = j.smtp_port || "—";
}

async function saveReceiver() {
  const email = $("alertTo").value.trim();
  if (!email) return toast("Email is required", false);

  const r = await fetch("/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ alert_to: email })
  });

  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    return toast(j.error || "Failed to save receiver", false);
  }
  toast(`Saved receiver: ${j.alert_to}`, true);
}

async function pollStatus() {
  try {
    const r = await fetch("/status", { cache: "no-store" });
    const j = await r.json();

    // badges (we don't have detection count in status, so we keep it simple)
    $("bFall").textContent = `Fall: ${j.fall_confirmed ? "YES" : "NO"}`;
    $("bSit").textContent = `Sitting: ${j.sit_too_long ? "YES" : "NO"}`;

    if (j.fall_confirmed) setPill("bad");
    else if (j.sit_too_long) setPill("warn");
    else setPill("ok");
  } catch (e) {
    setPill("idle");
  }
}

function setupTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      btn.classList.add("active");

      const target = btn.dataset.tab;
      document.querySelectorAll(".tab-body").forEach(b => b.classList.remove("active"));
      document.getElementById(`tab-${target}`).classList.add("active");
    });
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  setupTabs();

  $("btnSaveReceiver").addEventListener("click", saveReceiver);
  $("btnReloadSettings").addEventListener("click", async () => {
    await loadSettings();
    toast("Settings reloaded");
  });

  $("btnTestToast").addEventListener("click", () => toast("UI is working ✅"));

  await loadSettings();
  pollStatus();
  setInterval(pollStatus, 1000);
});

