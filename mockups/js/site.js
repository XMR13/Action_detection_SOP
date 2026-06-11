(function () {
  const apiFetchJson = async (path, options) => {
    const res = await fetch(path, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", ...(options && options.headers ? options.headers : {}) },
      ...options,
    });
    if (res.status === 401) {
      const body = document.body;
      const onLogin = body && body.classList.contains("page-login");
      if (!onLogin) {
        const herePath = window.location.pathname || "";
        const underUi = herePath.startsWith("/ui/") ? herePath.slice("/ui/".length) : "";
        const next = `${underUi || ""}${window.location.search || ""}${window.location.hash || ""}`;
        const nextParam = next ? `?next=${encodeURIComponent(next)}` : "";
        window.location.assign(`login.html${nextParam}`);
      }
      throw new Error("Unauthorized");
    }
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status} ${res.statusText}${text ? `: ${text}` : ""}`);
    }
    return await res.json();
  };

  const displayStepStatus = (raw) => {
    if (!raw) return "-";
    return String(raw).replaceAll("_", " ").toUpperCase();
  };

  const escapeHtml = (raw) =>
    String(raw ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const pillClassForStepStatus = (raw) => {
    const v = String(raw || "").toUpperCase();
    if (v === "DONE") return "yes";
    if (v === "NOT_DONE") return "no";
    if (v === "UNKNOWN") return "dir-b";
    return "";
  };

  const displayReviewStatus = (raw) => {
    const v = String(raw || "PENDING").toUpperCase();
    if (v === "QUALIFIED") return "QUALIFIED";
    if (v === "NOT_QUALIFIED") return "NOT QUALIFIED";
    return "PENDING";
  };

  const pillClassForReviewStatus = (raw) => {
    const v = String(raw || "PENDING").toUpperCase();
    if (v === "QUALIFIED") return "yes";
    if (v === "NOT_QUALIFIED") return "no";
    return "pending";
  };

  const structuredSop = (row) => {
    const sop = row && row.sop && typeof row.sop === "object" ? row.sop : null;
    return sop && sop.profile ? sop : null;
  };

  const sopScope = (row, scope) => {
    const sop = structuredSop(row);
    const data = sop && sop[scope] && typeof sop[scope] === "object" ? sop[scope] : null;
    return data || {};
  };

  const sopStatusValue = (row, scope) => {
    const data = sopScope(row, scope);
    const fallback =
      scope === "final"
        ? row && (row.final_sop || row.final_helmet || row.machine_sop || row.machine_helmet)
        : row && (row.machine_sop || row.machine_helmet);
    return String(data.status || fallback || "UNKNOWN").toUpperCase();
  };

  const rollOverallDisplay = (raw) => {
    const v = String(raw || "UNKNOWN").toUpperCase();
    if (v === "SESUAI SOP") return "Sesuai SOP";
    if (v === "TIDAK SESUAI SOP") return "Tidak sesuai SOP";
    return "Unknown";
  };

  const sopQueueSummary = (row, scope) => {
    const sop = structuredSop(row);
    if (!sop || sop.profile !== "roll_sop_v1") return "";
    const data = sopScope(row, scope);
    return [
      `Cleaned ${displayStepStatus(data.cleaned || "UNKNOWN")}`,
      `Labeled ${displayStepStatus(data.labeled || "UNKNOWN")}`,
    ].join(" · ");
  };

  const queueDecision = (row) => {
    const machine = sopStatusValue(row, "machine");
    const final = sopStatusValue(row, "final");
    const review = String((row && row.review_status) || "PENDING").toUpperCase();
    const summary = sopQueueSummary(row, "final") || sopQueueSummary(row, "machine");

    if (review === "PENDING") {
      return {
        label: "Pending review",
        className: "pending",
        meta: summary || `AI ${displayStepStatus(machine)}`,
      };
    }

    if (review === "NOT_QUALIFIED") {
      const changed = machine !== final;
      return {
        label: "Rejected",
        className: "no",
        meta:
          final === "DONE"
            ? `Reviewer rejected; AI ${displayStepStatus(machine)}`
            : changed
            ? `AI ${displayStepStatus(machine)} -> Final ${displayStepStatus(final)}`
            : summary || `AI ${displayStepStatus(machine)}`,
      };
    }

    const changed = machine !== final;
    return {
      label: displayStepStatus(final),
      className: pillClassForStepStatus(final),
      meta: changed ? `AI ${displayStepStatus(machine)} -> Final ${displayStepStatus(final)}` : summary || `AI ${displayStepStatus(machine)}`,
    };
  };

  const dashboardSopPills = (session) => {
    const sop = structuredSop(session);
    if (sop && sop.profile === "roll_sop_v1") {
      const final = sopScope(session, "final");
      const cleaned = String(final.cleaned || "UNKNOWN");
      const labeled = String(final.labeled || "UNKNOWN");
      const finalStatus = sopStatusValue(session, "final");
      return [
        `<span class="pill ${pillClassForStepStatus(cleaned)}">Cleaned ${displayStepStatus(cleaned)}</span>`,
        `<span class="pill ${pillClassForStepStatus(labeled)}">Labeled ${displayStepStatus(labeled)}</span>`,
        `<span class="pill ${pillClassForStepStatus(finalStatus)}">Final ${displayStepStatus(finalStatus)}</span>`,
      ].join(" ");
    }
    const roi = String(session.machine_roi_dwell || "UNKNOWN");
    const sopStatus = String(session.final_sop || session.machine_sop || "UNKNOWN");
    return `<span class="pill ${pillClassForStepStatus(roi)}">ROI ${displayStepStatus(roi)}</span> <span class="pill ${pillClassForStepStatus(sopStatus)}">SOP ${displayStepStatus(sopStatus)}</span>`;
  };

  const renderOverrideSelect = ({ key, label, machineValue, finalValue, overrideValue, values, displayValue }) => {
    const options = [
      `<option value="">Ikuti AI (${escapeHtml(displayValue(machineValue))})</option>`,
      ...values.map((value) => {
        const selected = String(overrideValue || "") === String(value) ? " selected" : "";
        return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(displayValue(value))}</option>`;
      }),
    ].join("");
    return `
      <div class="sop-override-row">
        <div>
          <span class="sop-override-label">${escapeHtml(label)}</span>
          <span class="sop-override-machine">AI: ${escapeHtml(displayValue(machineValue))}</span>
          <span class="sop-override-final">Final: ${escapeHtml(displayValue(finalValue))}</span>
        </div>
        <select data-override-key="${escapeHtml(key)}" aria-label="${escapeHtml(label)} override">
          ${options}
        </select>
      </div>
    `;
  };

  const renderSopPanel = (payload) => {
    const panel = document.getElementById("detail-sop-panel");
    if (!(panel instanceof HTMLElement)) return;

    const reviewOverrides =
      payload && payload.review && payload.review.overrides && typeof payload.review.overrides === "object"
        ? payload.review.overrides
        : {};
    const sop = structuredSop(payload);

    if (sop && sop.profile === "roll_sop_v1") {
      const machine = sopScope(payload, "machine");
      const final = sopScope(payload, "final");
      const rows = [
        renderOverrideSelect({
          key: "cleaned",
          label: "Cleaned",
          machineValue: machine.cleaned || "UNKNOWN",
          finalValue: final.cleaned || "UNKNOWN",
          overrideValue: reviewOverrides.cleaned || "",
          values: ["DONE", "NOT_DONE", "UNKNOWN"],
          displayValue: displayStepStatus,
        }),
        renderOverrideSelect({
          key: "labeled",
          label: "Labeled",
          machineValue: machine.labeled || "UNKNOWN",
          finalValue: final.labeled || "UNKNOWN",
          overrideValue: reviewOverrides.labeled || "",
          values: ["DONE", "NOT_DONE", "UNKNOWN"],
          displayValue: displayStepStatus,
        }),
        renderOverrideSelect({
          key: "overall_status",
          label: "Final status",
          machineValue: machine.overall_status || "UNKNOWN",
          finalValue: final.overall_status || "UNKNOWN",
          overrideValue: reviewOverrides.overall_status || "",
          values: ["SESUAI SOP", "TIDAK SESUAI SOP", "UNKNOWN"],
          displayValue: rollOverallDisplay,
        }),
      ].join("");
      const inconsistent = sop.inconsistent
        ? `<span class="pill no">AI overall berbeda dari cleaned/labeled</span>`
        : `<span class="pill yes">Roll SOP konsisten</span>`;
      panel.innerHTML = `
        <div class="detail-sop-head">
          <div>
            <strong>Roll SOP</strong>
            <p class="caption">Override hanya jika bukti menunjukkan hasil AI perlu dikoreksi.</p>
          </div>
          ${inconsistent}
        </div>
        <div class="sop-override-grid">${rows}</div>
      `;
      return;
    }

    const machine = sopScope(payload, "machine");
    const final = sopScope(payload, "final");
    const legacyRows = [
      renderOverrideSelect({
        key: "operator_present",
        label: "Operator present",
        machineValue: machine.operator_present || payload.machine_operator || "UNKNOWN",
        finalValue: final.operator_present || "UNKNOWN",
        overrideValue: reviewOverrides.operator_present || "",
        values: ["DONE", "NOT_DONE", "UNKNOWN"],
        displayValue: displayStepStatus,
      }),
      renderOverrideSelect({
        key: "roi_dwell",
        label: "ROI dwell",
        machineValue: machine.roi_dwell || payload.machine_roi_dwell || "UNKNOWN",
        finalValue: final.roi_dwell || "UNKNOWN",
        overrideValue: reviewOverrides.roi_dwell || "",
        values: ["DONE", "NOT_DONE", "UNKNOWN"],
        displayValue: displayStepStatus,
      }),
      renderOverrideSelect({
        key: "helmet",
        label: "Helmet",
        machineValue: machine.helmet || payload.machine_helmet || "UNKNOWN",
        finalValue: final.helmet || payload.final_helmet || "UNKNOWN",
        overrideValue: reviewOverrides.helmet || "",
        values: ["DONE", "NOT_DONE", "UNKNOWN"],
        displayValue: displayStepStatus,
      }),
    ].join("");
    panel.innerHTML = `
      <div class="detail-sop-head">
        <div>
          <strong>Legacy operator MVP-A</strong>
          <p class="caption">Kompatibilitas untuk sesi lama.</p>
        </div>
      </div>
      <div class="sop-override-grid">${legacyRows}</div>
    `;
  };

  const formatHmsFromIso = (iso) => {
    if (!iso) return "-";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleTimeString("en-GB", { hour12: false });
  };

  const formatDateTimeFromIso = (iso) => {
    if (!iso) return "-";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString("en-GB", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  };

  const normalizeShiftId = (raw) => {
    const key = String(raw || "")
      .trim()
      .toUpperCase()
      .replaceAll(" ", "")
      .replaceAll("_", "");
    if (key === "S1" || key === "SHIFT1" || key === "1") return "S1";
    if (key === "S2" || key === "SHIFT2" || key === "2") return "S2";
    if (key === "S3" || key === "SHIFT3" || key === "3") return "S3";
    return "";
  };

  const shiftLabel = (shiftId, shiftName) => {
    const explicit = String(shiftName || "").trim();
    if (explicit) return explicit;
    const sid = normalizeShiftId(shiftId);
    if (sid === "S1") return "Shift 1";
    if (sid === "S2") return "Shift 2";
    if (sid === "S3") return "Shift 3";
    return "-";
  };

  const formatDuration = (seconds) => {
    const s = Math.max(0, Number(seconds || 0));
    const mm = Math.floor(s / 60);
    const ss = s - mm * 60;
    const mmStr = String(mm).padStart(2, "0");
    const ssStr = ss.toFixed(1).padStart(4, "0");
    return `${mmStr}:${ssStr}`;
  };

  const formatHmLocal = (d) => {
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    return `${hh}:${mm}`;
  };

  const buildSmoothPath = (points) => {
    if (!Array.isArray(points) || points.length < 2) return "";
    const p = points.map((pt) => ({ x: Number(pt.x), y: Number(pt.y) }));
    if (p.some((pt) => Number.isNaN(pt.x) || Number.isNaN(pt.y))) return "";
    let d = `M${p[0].x.toFixed(1)} ${p[0].y.toFixed(1)}`;
    for (let i = 0; i < p.length - 1; i++) {
      const p0 = i > 0 ? p[i - 1] : p[i];
      const p1 = p[i];
      const p2 = p[i + 1];
      const p3 = i + 2 < p.length ? p[i + 2] : p2;
      const c1x = p1.x + (p2.x - p0.x) / 6;
      const c1y = p1.y + (p2.y - p0.y) / 6;
      const c2x = p2.x - (p3.x - p1.x) / 6;
      const c2y = p2.y - (p3.y - p1.y) / 6;
      d += ` C${c1x.toFixed(1)} ${c1y.toFixed(1)},${c2x.toFixed(1)} ${c2y.toFixed(1)},${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
    }
    return d;
  };

  const DATE_YMD_RE = /^\d{4}-\d{2}-\d{2}$/;

  const isValidDateYmd = (raw) => DATE_YMD_RE.test(String(raw || ""));

  const toYmdLocal = (d) => {
    const yy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    return `${yy}-${mm}-${dd}`;
  };

  const readDateSliceFromUrl = () => {
    const params = new URLSearchParams(window.location.search || "");
    const exact = params.get("date");
    if (isValidDateYmd(exact)) {
      return { from: String(exact), to: String(exact) };
    }
    const from = params.get("date_from");
    const to = params.get("date_to");
    return {
      from: isValidDateYmd(from) ? String(from) : "",
      to: isValidDateYmd(to) ? String(to) : "",
    };
  };

  const writeDateSliceToUrl = (slice, onChange) => {
    const from = slice && isValidDateYmd(slice.from) ? String(slice.from) : "";
    const to = slice && isValidDateYmd(slice.to) ? String(slice.to) : "";
    if (from && to && from > to) {
      alert("Invalid date range: date_from is after date_to.");
      return;
    }
    const url = new URL(window.location.href);
    url.searchParams.delete("date");
    url.searchParams.delete("date_from");
    url.searchParams.delete("date_to");
    if (from) url.searchParams.set("date_from", from);
    if (to) url.searchParams.set("date_to", to);
    if (typeof window.history.replaceState === "function") {
      window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
    }
    if (typeof onChange === "function") onChange();
  };

  const dateSliceLabel = () => {
    const slice = readDateSliceFromUrl();
    if (slice.from && slice.to) {
      if (slice.from === slice.to) return `Date: ${slice.from}`;
      return `Date: ${slice.from} -> ${slice.to}`;
    }
    if (slice.from) return `Date: from ${slice.from}`;
    if (slice.to) return `Date: until ${slice.to}`;
    return "Date: all";
  };

  const buildDateApiQuery = () => {
    const slice = readDateSliceFromUrl();
    const params = new URLSearchParams();
    if (slice.from) params.set("date_from", slice.from);
    if (slice.to) params.set("date_to", slice.to);
    return params.toString();
  };

  const withDateApiQuery = (url) => {
    const query = buildDateApiQuery();
    if (!query) return url;
    return `${url}${url.includes("?") ? "&" : "?"}${query}`;
  };

  const buildUiHrefWithDate = (page, hashRaw) => {
    const slice = readDateSliceFromUrl();
    const params = new URLSearchParams();
    if (slice.from) params.set("date_from", slice.from);
    if (slice.to) params.set("date_to", slice.to);
    const query = params.toString();
    const hash = hashRaw ? `#${hashRaw}` : "";
    return `${page}${query ? `?${query}` : ""}${hash}`;
  };

  const buildSessionDetailHref = (sessionUid) => {
    const slice = readDateSliceFromUrl();
    const params = new URLSearchParams();
    if (slice.from) params.set("date_from", slice.from);
    if (slice.to) params.set("date_to", slice.to);
    if (sessionUid) params.set("session_uid", String(sessionUid));
    const query = params.toString();
    const hash = sessionUid ? `#${encodeURIComponent(String(sessionUid))}` : "";
    return `session-detail.html${query ? `?${query}` : ""}${hash}`;
  };

  const readSessionUidFromUrl = () => {
    const params = new URLSearchParams(window.location.search || "");
    const fromQuery = params.get("session_uid");
    if (fromQuery) return String(fromQuery);
    const hash = window.location.hash ? window.location.hash.slice(1) : "";
    return hash ? decodeURIComponent(hash) : "";
  };

  const applyDateSliceToStaticNav = () => {
    const navTargets = new Set(["index.html", "review-queue.html", "session-detail.html", "setup.html"]);
    document.querySelectorAll("a[href]").forEach((node) => {
      if (!(node instanceof HTMLAnchorElement)) return;
      const href = node.getAttribute("href") || "";
      if (!href || href.startsWith("#") || href.startsWith("http://") || href.startsWith("https://") || href.startsWith("mailto:")) {
        return;
      }
      const u = new URL(href, window.location.href);
      const path = u.pathname.split("/").pop() || "";
      if (!navTargets.has(path)) return;
      const hash = u.hash ? u.hash.slice(1) : "";
      node.setAttribute("href", buildUiHrefWithDate(path, hash));
    });
  };

  const bindDateControls = ({
    fromId,
    toId,
    rangeId,
    applyId,
    clearId,
    labelId,
    onChange,
  }) => {
    const fromInput = document.getElementById(fromId);
    const toInput = document.getElementById(toId);
    const rangeSel = rangeId ? document.getElementById(rangeId) : null;
    const applyBtn = document.getElementById(applyId);
    const clearBtn = document.getElementById(clearId);
    const label = document.getElementById(labelId);

    const refreshUi = () => {
      const slice = readDateSliceFromUrl();
      if (fromInput instanceof HTMLInputElement) fromInput.value = slice.from || "";
      if (toInput instanceof HTMLInputElement) toInput.value = slice.to || "";
      if (label) label.textContent = dateSliceLabel();
      if (rangeSel instanceof HTMLSelectElement) {
        const today = toYmdLocal(new Date());
        const day7 = new Date();
        day7.setDate(day7.getDate() - 6);
        const last7 = toYmdLocal(day7);
        const day30 = new Date();
        day30.setDate(day30.getDate() - 29);
        const last30 = toYmdLocal(day30);
        let val = "CUSTOM";
        if (slice.from === today && slice.to === today) val = "TODAY";
        else if (slice.from === last7 && slice.to === today) val = "LAST_7_DAYS";
        else if (slice.from === last30 && slice.to === today) val = "LAST_30_DAYS";
        rangeSel.value = val;
      }
    };

    const applyFromInputs = () => {
      let from = fromInput instanceof HTMLInputElement ? String(fromInput.value || "") : "";
      let to = toInput instanceof HTMLInputElement ? String(toInput.value || "") : "";
      // UX rule: one selected bound means exact-date filter (most expected behavior for reviewers).
      if (from && !to) to = from;
      if (to && !from) from = to;
      const next = { from, to };
      writeDateSliceToUrl(next, () => {
        applyDateSliceToStaticNav();
        refreshUi();
        if (typeof onChange === "function") onChange();
      });
    };

    if (applyBtn instanceof HTMLButtonElement) {
      applyBtn.addEventListener("click", applyFromInputs);
    }
    if (clearBtn instanceof HTMLButtonElement) {
      clearBtn.addEventListener("click", () => {
        writeDateSliceToUrl({ from: "", to: "" }, () => {
          applyDateSliceToStaticNav();
          refreshUi();
          if (typeof onChange === "function") onChange();
        });
      });
    }
    if (rangeSel instanceof HTMLSelectElement) {
      rangeSel.addEventListener("change", () => {
        const today = toYmdLocal(new Date());
        let from = "";
        let to = "";
        const choice = String(rangeSel.value || "CUSTOM");
        if (choice === "TODAY") {
          from = today;
          to = today;
        } else if (choice === "LAST_7_DAYS") {
          const d = new Date();
          d.setDate(d.getDate() - 6);
          from = toYmdLocal(d);
          to = today;
        } else if (choice === "LAST_30_DAYS") {
          const d = new Date();
          d.setDate(d.getDate() - 29);
          from = toYmdLocal(d);
          to = today;
        }
        if (fromInput instanceof HTMLInputElement) fromInput.value = from;
        if (toInput instanceof HTMLInputElement) toInput.value = to;
        if (choice !== "CUSTOM") applyFromInputs();
      });
    }

    refreshUi();
  };

  const lockFormOnSubmit = (form) => {
    form.addEventListener("submit", (event) => {
      if (event.defaultPrevented) return;
      form.classList.add("is-submitting");
      const buttons = form.querySelectorAll("button");
      buttons.forEach((button) => {
        button.setAttribute("disabled", "disabled");
      });
    });
  };

  document.querySelectorAll("form[data-disable-on-submit]").forEach((formNode) => {
    if (formNode instanceof HTMLFormElement) {
      lockFormOnSubmit(formNode);
    }
  });

  const initAuthUi = () => {
    const logoutLink = document.querySelector(".nav-logout");
    if (logoutLink instanceof HTMLAnchorElement) {
      logoutLink.addEventListener("click", async (event) => {
        event.preventDefault();
        try {
          await apiFetchJson("/api/auth/logout", { method: "POST" });
        } catch (err) {
          // ignore
        } finally {
          window.location.assign("login.html");
        }
      });
    }

    const body = document.body;
    if (!body || !body.classList.contains("page-login")) {
      return;
    }

    const loginForm = document.getElementById("login-form");
    if (!(loginForm instanceof HTMLFormElement)) {
      return;
    }

    const usernameInput = document.getElementById("username");
    const passwordInput = document.getElementById("password");
    const statusBox = document.getElementById("login-status");
    const submitBtn = loginForm.querySelector("button[type='submit']");

    const setStatus = (text, kind) => {
      if (!(statusBox instanceof HTMLElement)) return;
      const cls = kind === "error" ? "validation-summary no" : kind === "ok" ? "validation-summary yes" : "validation-summary";
      statusBox.className = cls;
      statusBox.innerHTML = `<p>${text}</p>`;
    };

    loginForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const username = usernameInput instanceof HTMLInputElement ? String(usernameInput.value || "").trim() : "";
      const password = passwordInput instanceof HTMLInputElement ? String(passwordInput.value || "") : "";
      if (!username || !password) {
        setStatus("Enter username and password.", "error");
        return;
      }
      if (submitBtn instanceof HTMLButtonElement) submitBtn.setAttribute("disabled", "disabled");
      setStatus("Signing in...", "ok");
      try {
        await apiFetchJson("/api/auth/login", {
          method: "POST",
          body: JSON.stringify({ username, password }),
        });
        const params = new URLSearchParams(window.location.search || "");
        const next = params.get("next");
        window.location.assign(next && !next.startsWith("http") ? String(next) : "index.html");
      } catch (err) {
        setStatus("Login failed. Check your credentials.", "error");
        if (submitBtn instanceof HTMLButtonElement) submitBtn.removeAttribute("disabled");
      }
    });
  };

  initAuthUi();

  const formNode = document.getElementById("review-form");
  const form = formNode instanceof HTMLFormElement ? formNode : null;
  const statusInput = document.getElementById("review-status");
  const actionButtons = form ? form.querySelectorAll("button[data-review-status]") : [];
  let queueLinks = [];
  const selectedSessionIdInput = document.getElementById("selected-session-id");
  const selectedSessionInline = document.getElementById("selected-session-inline");
  const selectedSessionLabel = document.getElementById("selected-session-label");
  const selectedSessionUid = document.getElementById("selected-session-uid");
  const selectedSessionDate = document.getElementById("selected-session-date");
  const selectedSessionShift = document.getElementById("selected-session-shift");
  const selectedMachineStatus = document.getElementById("selected-machine-status");
  const selectedHumanStatus = document.getElementById("selected-human-status");
  const selectedFinalStatus = document.getElementById("selected-final-status");
  const selectedEvidence = document.getElementById("selected-evidence");
  const selectedDetailLink = document.getElementById("selected-detail-link");
  const queueOpenSelected = document.getElementById("queue-open-selected");
  let queuePage = 1;
  let queuePageSize = 20;

  const queuePaginationMeta = document.getElementById("queue-page-meta");
  const queuePaginationIndicator = document.getElementById("queue-page-indicator");
  const queuePagePrevBtn = document.getElementById("queue-page-prev");
  const queuePageNextBtn = document.getElementById("queue-page-next");

  const readQueuePageSize = () => {
    const node = document.getElementById("queue-page-size");
    const raw = node instanceof HTMLSelectElement ? Number(node.value) : Number.NaN;
    if (raw === 10 || raw === 20) return raw;
    return 20;
  };

  const resetQueuePage = () => {
    queuePage = 1;
  };

  const syncQueuePaginationUi = ({ total, page, pageSize, totalPages, hasPrev, hasNext }) => {
    const safeTotal = Math.max(0, Number(total || 0));
    const safePage = Math.max(1, Number(page || 1));
    const safePageSize = Math.max(1, Number(pageSize || 15));
    const safePages = Math.max(0, Number(totalPages || 0));
    const from = safeTotal > 0 ? (safePage - 1) * safePageSize + 1 : 0;
    const to = safeTotal > 0 ? Math.min(safeTotal, safePage * safePageSize) : 0;

    if (queuePaginationMeta) {
      queuePaginationMeta.textContent = safeTotal > 0 ? `Showing ${from}-${to} of ${safeTotal}` : "Showing 0";
    }
    if (queuePaginationIndicator) {
      queuePaginationIndicator.textContent = safePages > 0 ? `Page ${safePage}/${safePages}` : "Page 0/0";
    }

    if (queuePagePrevBtn instanceof HTMLButtonElement) {
      queuePagePrevBtn.disabled = !hasPrev;
      queuePagePrevBtn.setAttribute("aria-disabled", hasPrev ? "false" : "true");
    }
    if (queuePageNextBtn instanceof HTMLButtonElement) {
      queuePageNextBtn.disabled = !hasNext;
      queuePageNextBtn.setAttribute("aria-disabled", hasNext ? "false" : "true");
    }
  };

  const syncReviewDock = (link) => {
    const row = link.closest("tr");
    if (!(row instanceof HTMLTableRowElement)) {
      return;
    }

    const { sessionId, sessionUid, date, shift, machine, human, final, evidence } = row.dataset;
    const resolvedSessionId = sessionId || link.textContent?.trim() || "UNKNOWN";
    const resolvedShift = shift || "-";
    const resolvedMachine = machine || "-";
    const resolvedHuman = human || "-";
    const resolvedFinal = final || "-";
    const resolvedEvidence = evidence || "-";
    const resolvedSessionUid = sessionUid || "-";
    const resolvedDate = date || "-";
    if (selectedSessionIdInput instanceof HTMLInputElement) {
      selectedSessionIdInput.value = resolvedSessionId;
    }
    if (selectedSessionInline) {
      selectedSessionInline.textContent = resolvedSessionId;
    }
    if (selectedSessionLabel) {
      selectedSessionLabel.textContent = `${resolvedSessionId} | ${resolvedDate} | ${resolvedShift} | Machine ${resolvedMachine}`;
    }
    if (selectedSessionUid) {
      selectedSessionUid.textContent = resolvedSessionUid;
    }
    if (selectedSessionDate) {
      selectedSessionDate.textContent = resolvedDate;
    }
    if (selectedSessionShift) {
      selectedSessionShift.textContent = resolvedShift;
    }
    if (selectedMachineStatus) {
      selectedMachineStatus.textContent = resolvedMachine;
    }
    if (selectedHumanStatus) {
      selectedHumanStatus.textContent = resolvedHuman;
    }
    if (selectedFinalStatus) {
      selectedFinalStatus.textContent = resolvedFinal;
    }
    if (selectedEvidence) {
      selectedEvidence.textContent = resolvedEvidence;
    }
    if (selectedDetailLink instanceof HTMLAnchorElement) {
      const targetUid = resolvedSessionUid && resolvedSessionUid !== "-" ? resolvedSessionUid : resolvedSessionId;
      selectedDetailLink.href = buildSessionDetailHref(String(targetUid));
    }
    if (queueOpenSelected instanceof HTMLAnchorElement) {
      const targetUid = resolvedSessionUid && resolvedSessionUid !== "-" ? resolvedSessionUid : resolvedSessionId;
      queueOpenSelected.href = buildSessionDetailHref(String(targetUid));
    }
  };

  const updateHashFromLink = (link) => {
    const href = link.getAttribute("href") || "";
    if (!href.startsWith("#")) {
      return false;
    }

    if (typeof window.history.replaceState === "function") {
      window.history.replaceState(null, "", href);
      return true;
    }

    window.location.hash = href.slice(1);
    return true;
  };

  const setActiveQueueIndex = (index) => {
    let activeLink = null;
    queueLinks.forEach((link, queueIndex) => {
      const isActive = queueIndex === index;
      link.classList.toggle("active", isActive);
      const row = link.closest("tr");
      if (row) {
        row.classList.toggle("queue-row-active", isActive);
      }
      if (isActive) {
        activeLink = link;
      }
    });

    if (activeLink) {
      syncReviewDock(activeLink);
    }
  };

  const submitWithStatus = (status) => {
    if (!form) {
      return;
    }

    if (statusInput instanceof HTMLInputElement) {
      statusInput.value = status;
    }

    if (typeof form.requestSubmit === "function") {
      form.requestSubmit();
      return;
    }

    form.submit();
  };

  const navigateQueue = (step) => {
    if (queueLinks.length === 0) {
      return;
    }

    const activeIndex = queueLinks.findIndex((link) => link.classList.contains("active"));
    const currentIndex = activeIndex >= 0 ? activeIndex : 0;
    const nextIndex = (currentIndex + step + queueLinks.length) % queueLinks.length;

    if (nextIndex !== currentIndex) {
      const nextLink = queueLinks[nextIndex];
      setActiveQueueIndex(nextIndex);
      if (!updateHashFromLink(nextLink)) {
        window.location.assign(nextLink.href);
      }
    }
  };

  const initQueueInteractions = () => {
    queueLinks = Array.from(document.querySelectorAll("[data-queue-link]"));

    queueLinks.forEach((link, index) => {
      link.addEventListener("click", (event) => {
        const href = link.getAttribute("href") || "";
        setActiveQueueIndex(index);
        if (href.startsWith("#")) {
          event.preventDefault();
          updateHashFromLink(link);
        }
      });
    });

    document.querySelectorAll(".queue-table tbody tr").forEach((rowNode) => {
      if (!(rowNode instanceof HTMLTableRowElement)) {
        return;
      }

      rowNode.addEventListener("click", (event) => {
        const target = event.target;
        if (target instanceof Element && target.closest("a, button, input, select, textarea, label")) {
          return;
        }

        const rowLink = rowNode.querySelector("[data-queue-link]");
        if (!(rowLink instanceof HTMLAnchorElement)) {
          return;
        }

        const rowIndex = queueLinks.indexOf(rowLink);
        if (rowIndex < 0) {
          return;
        }

        setActiveQueueIndex(rowIndex);
        updateHashFromLink(rowLink);
      });
    });

    if (queueLinks.length > 0) {
      const activeIndex = queueLinks.findIndex((link) => link.classList.contains("active"));
      const hashIndex = queueLinks.findIndex((link) => (link.getAttribute("href") || "") === window.location.hash);
      const initialIndex = hashIndex >= 0 ? hashIndex : activeIndex >= 0 ? activeIndex : 0;
      setActiveQueueIndex(initialIndex);
    }
  };

  actionButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const status = button.getAttribute("data-review-status") || "PENDING";
      if (statusInput instanceof HTMLInputElement) {
        statusInput.value = status;
      }
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) {
      return;
    }

    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) {
      return;
    }

    const key = event.key.toLowerCase();
    if (key === "y") {
      if (!form) {
        return;
      }
      event.preventDefault();
      submitWithStatus("QUALIFIED");
      return;
    }

    if (key === "n") {
      if (!form) {
        return;
      }
      event.preventDefault();
      submitWithStatus("NOT_QUALIFIED");
      return;
    }

    if (key === "s") {
      if (!form) {
        return;
      }
      event.preventDefault();
      submitWithStatus("PENDING");
      return;
    }

    if (key === "j") {
      if (queueLinks.length === 0) {
        return;
      }
      event.preventDefault();
      navigateQueue(-1);
      return;
    }

    if (key === "k") {
      if (queueLinks.length === 0) {
        return;
      }
      event.preventDefault();
      navigateQueue(1);
      return;
    }

  });

  const populateQueue = async () => {
    const tbody = document.querySelector(".queue-table tbody");
    const queueBody = document.body;
    if (!(tbody instanceof HTMLTableSectionElement)) {
      return;
    }
    if (queueBody && queueBody.classList.contains("page-review-queue")) {
      queueBody.classList.add("is-hydrating");
    }
    const queueDateLabel = document.getElementById("queue-active-date-slice");
    if (queueDateLabel) queueDateLabel.textContent = dateSliceLabel();

    // Sync summary counters on the queue page (topbar + cards).
    try {
      const stats = await apiFetchJson(withDateApiQuery("/api/stats"));
      const pending = stats && stats.pending != null ? Number(stats.pending) : null;
      const approved = stats && stats.approved != null ? Number(stats.approved) : null;
      const rejected = stats && stats.rejected != null ? Number(stats.rejected) : null;
      const unknown = stats && stats.unknown != null ? Number(stats.unknown) : null;

      const queueLengthHint = document.getElementById("queue-length-hint");
      if (queueLengthHint && pending != null && !Number.isNaN(pending)) {
        queueLengthHint.textContent = `Jumlah antrian: ${pending}`;
      }
      const pendingPill = document.getElementById("queue-pending-pill");
      if (pendingPill && pending != null && !Number.isNaN(pending)) {
        pendingPill.textContent = `menunggu ${pending}`;
      }
      const pendingNode = document.getElementById("queue-stat-pending");
      if (pendingNode && pending != null && !Number.isNaN(pending)) {
        pendingNode.textContent = String(pending);
      }
      const approvedNode = document.getElementById("queue-stat-approved");
      if (approvedNode && approved != null && !Number.isNaN(approved)) {
        approvedNode.textContent = String(approved);
      }
      const rejectedNode = document.getElementById("queue-stat-rejected");
      if (rejectedNode && rejected != null && !Number.isNaN(rejected)) {
        rejectedNode.textContent = String(rejected);
      }
      const unknownNode = document.getElementById("queue-stat-unknown");
      if (unknownNode && unknown != null && !Number.isNaN(unknown)) {
        unknownNode.textContent = String(unknown);
      }
    } catch (err) {
      // ignore stats failures; queue list still loads
    }

    const statusSel = document.getElementById("queue-status");
    const evidenceSel = document.getElementById("queue-evidence");
    const shiftSel = document.getElementById("queue-shift");
    const sortSel = document.getElementById("queue-sort");
    const pageSizeSel = document.getElementById("queue-page-size");

    const reviewStatus = statusSel instanceof HTMLSelectElement ? String(statusSel.value || "") : "";
    const evidenceFilter = evidenceSel instanceof HTMLSelectElement ? String(evidenceSel.value || "ANY") : "ANY";
    const shiftFilter = shiftSel instanceof HTMLSelectElement ? String(shiftSel.value || "ALL") : "ALL";
    const sort = sortSel instanceof HTMLSelectElement ? String(sortSel.value || "NEWEST") : "NEWEST";
    queuePageSize = pageSizeSel instanceof HTMLSelectElement ? readQueuePageSize() : queuePageSize;

    let url = withDateApiQuery(
      `/api/sessions?page=${encodeURIComponent(String(queuePage))}&page_size=${encodeURIComponent(
        String(queuePageSize)
      )}&sort=${encodeURIComponent(sort || "NEWEST")}&evidence=${encodeURIComponent(evidenceFilter || "ANY")}&shift=${encodeURIComponent(
        shiftFilter || "ALL"
      )}`
    );
    if (reviewStatus && reviewStatus !== "ALL") {
      url += `&review_status=${encodeURIComponent(reviewStatus)}`;
    }

    let payload;
    try {
      payload = await apiFetchJson(url);
    } catch (err) {
      syncQueuePaginationUi({ total: 0, page: 1, pageSize: queuePageSize, totalPages: 0, hasPrev: false, hasNext: false });
      tbody.innerHTML = `<tr><td colspan="8"><span class="pill no">Failed to load sessions</span></td></tr>`;
      if (queueBody && queueBody.classList.contains("page-review-queue")) {
        queueBody.classList.remove("is-hydrating");
      }
      return;
    }

    let sessions = Array.isArray(payload.sessions) ? payload.sessions : [];
    const serverTotalPages = Number(payload.total_pages || 0);
    if (serverTotalPages > 0 && queuePage > serverTotalPages) {
      queuePage = serverTotalPages;
      await populateQueue();
      return;
    }
    syncQueuePaginationUi({
      total: Number(payload.total || sessions.length),
      page: Number(payload.page || queuePage),
      pageSize: Number(payload.page_size || queuePageSize),
      totalPages: serverTotalPages,
      hasPrev: Boolean(payload.has_prev),
      hasNext: Boolean(payload.has_next),
    });

    if (sessions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8"><span class="pill">No sessions found</span></td></tr>`;
      if (queueBody && queueBody.classList.contains("page-review-queue")) {
        queueBody.classList.remove("is-hydrating");
      }
      return;
    }

    const rowsHtml = sessions
      .map((s, index) => {
        const uid = String(s.session_uid || "");
        const rawSid = String(s.session_id || "").trim();
        const sid = rawSid || (uid ? `Session ${String((queuePage - 1) * queuePageSize + index + 1).padStart(2, "0")}` : "-");
        const machine = sopStatusValue(s, "machine");
        const final = sopStatusValue(s, "final");
        const decision = queueDecision(s);
        const date = String(s.date || "-");
        const shift = shiftLabel(s.shift_id, s.shift_name);
        const start = formatHmsFromIso(s.start_time_iso);
        const dur = formatDuration(s.duration_s);
        const thumbUrl = s.thumbnail_url
          ? String(s.thumbnail_url)
          : s.has_thumbnail
          ? `/media/${encodeURIComponent(uid)}/thumbnail.jpg`
          : "";
        const evidenceLabel =
          s.clip_count > 0 ? (s.has_thumbnail ? "thumbnail + klip" : "klip") : s.has_thumbnail ? "thumbnail" : "-";
        const evidenceBadge = s.clip_count > 0 ? `<span class="queue-evidence-badge" title="${s.clip_count} clip(s)">▶</span>` : "";

        const rowActive = index === 0 ? " queue-row-active" : "";
        const linkActive = index === 0 ? " active" : "";

        return `
          <tr class="${rowActive.trim()}"
            data-session-id="${escapeHtml(sid)}"
            data-session-uid="${escapeHtml(uid)}"
            data-date="${escapeHtml(date)}"
            data-shift="${escapeHtml(shift)}"
            data-machine="${displayStepStatus(machine)}"
            data-human="${escapeHtml(decision.label)}"
            data-final="${displayStepStatus(final)}"
            data-evidence="${escapeHtml(evidenceLabel)}"
          >
            <td>
              <a class="queue-session-link${linkActive}" href="#${encodeURIComponent(uid)}" title="${uid ? `UID ${escapeHtml(uid)}` : ""}" data-queue-link>${escapeHtml(sid)}</a>
            </td>
            <td>${escapeHtml(date)}</td>
            <td><span class="pill ink">${escapeHtml(shift)}</span></td>
            <td>${escapeHtml(start)}</td>
            <td>${escapeHtml(dur)}</td>
            <td>
              <div class="queue-decision-cell">
                <span class="pill ${decision.className}">${escapeHtml(decision.label)}</span>
                ${decision.meta ? `<div class="table-sub queue-decision-meta">${escapeHtml(decision.meta)}</div>` : ""}
              </div>
            </td>
            <td>
              <div class="queue-evidence-cell">
                <div class="queue-evidence-preview ${thumbUrl ? "" : "queue-evidence-empty"}">
                  ${thumbUrl ? `<img class="queue-evidence-thumb" alt="session evidence thumbnail" src="${thumbUrl}" loading="lazy" />` : `<span class="queue-evidence-none">No preview</span>`}
                  ${evidenceBadge}
                </div>
              </div>
            </td>
            <td><a class="btn btn-compact action-inspect" href="${buildSessionDetailHref(uid)}">Inspect</a></td>
          </tr>
        `;
      })
      .join("");

    tbody.innerHTML = rowsHtml;
    initQueueInteractions();
    if (queueBody && queueBody.classList.contains("page-review-queue")) {
      queueBody.classList.remove("is-hydrating");
    }
  };

  const populateDetail = async () => {
    const sessionUid = readSessionUidFromUrl();
    if (!sessionUid) {
      // If user opens Session Detail from the sidebar, pick the newest session.
      try {
        const list = await apiFetchJson(withDateApiQuery("/api/sessions?limit=1"));
        const sessions = Array.isArray(list.sessions) ? list.sessions : [];
        if (sessions.length > 0 && sessions[0].session_uid) {
          window.location.assign(buildSessionDetailHref(String(sessions[0].session_uid)));
          return;
        }
      } catch (err) {
        // ignore
      }
      const playerEmpty = document.querySelector(".detail-player");
      if (playerEmpty instanceof HTMLElement) {
        playerEmpty.innerHTML = '<div class="frame-overlay"><span class="pill">tidak ada sesi</span></div>';
      }
      return;
    }

    let payload;
    try {
      payload = await apiFetchJson(`/api/sessions/${encodeURIComponent(sessionUid)}`);
    } catch (err) {
      return;
    }

    const sessionId = String(payload.session_id || "-");
    const sessionDate = String(payload.date || "-");
    const startIso = payload.checklist && payload.checklist.start_time_iso ? String(payload.checklist.start_time_iso) : "";
    const resolvedShift = shiftLabel(
      payload && payload.shift_id ? payload.shift_id : payload && payload.checklist ? payload.checklist.shift_id : "",
      payload && payload.shift_name ? payload.shift_name : payload && payload.checklist ? payload.checklist.shift_name : ""
    );
    const dateHint = document.getElementById("detail-session-date-hint");
    const sidNode = document.getElementById("detail-session-id");
    if (sidNode) sidNode.textContent = sessionId;
    if (dateHint) {
      const dateHintValue = startIso ? formatDateTimeFromIso(startIso) : sessionDate;
      dateHint.textContent = `Session time: ${dateHintValue}`;
    }
    if (selectedSessionIdInput instanceof HTMLInputElement) {
      selectedSessionIdInput.value = sessionId;
    }

    const overviewSession = document.getElementById("detail-overview-session");
    const overviewDate = document.getElementById("detail-overview-date");
    const overviewShift = document.getElementById("detail-overview-shift");
    const overviewDuration = document.getElementById("detail-overview-duration");
    const overviewAi = document.getElementById("detail-overview-ai");
    const overviewStatusCard = document.getElementById("detail-overview-status-card");
    if (overviewSession) overviewSession.textContent = sessionId;
    if (overviewDate) overviewDate.textContent = startIso ? formatDateTimeFromIso(startIso) : sessionDate;
    if (overviewShift) overviewShift.textContent = resolvedShift || "-";
    const startS =
      payload.checklist && payload.checklist.start_time_s != null ? Number(payload.checklist.start_time_s) : Number.NaN;
    const endS =
      payload.checklist && payload.checklist.end_time_s != null ? Number(payload.checklist.end_time_s) : Number.NaN;
    if (!Number.isNaN(startS) && !Number.isNaN(endS) && endS >= startS) {
      const durationText = `${(endS - startS).toFixed(1)} detik`;
      if (overviewDuration) overviewDuration.textContent = durationText;
    } else if (overviewDuration) {
      overviewDuration.textContent = "-";
    }

    const queueCrumb = document.getElementById("detail-queue-link");
    const backLink = document.getElementById("detail-back-link");
    if (queueCrumb instanceof HTMLAnchorElement) {
      queueCrumb.href = buildUiHrefWithDate("review-queue.html", encodeURIComponent(String(sessionUid)));
    }
    if (backLink instanceof HTMLAnchorElement) {
      backLink.href = buildUiHrefWithDate("review-queue.html", encodeURIComponent(String(sessionUid)));
    }

    const resolveNextPending = async () => {
      try {
        const list = await apiFetchJson(withDateApiQuery("/api/sessions?review_status=PENDING&sort=NEWEST&limit=200"));
        const sessions = Array.isArray(list.sessions) ? list.sessions : [];
        if (sessions.length === 0) return null;
        const idx = sessions.findIndex((s) => String(s.session_uid || "") === String(sessionUid));
        if (idx >= 0) {
          const next = sessions[(idx + 1) % sessions.length];
          return next && next.session_uid ? String(next.session_uid) : null;
        }
        const first = sessions[0];
        return first && first.session_uid ? String(first.session_uid) : null;
      } catch (err) {
        return null;
      }
    };

    const nextLink = document.getElementById("detail-next-link");
    if (nextLink instanceof HTMLAnchorElement) {
      const nextUid = await resolveNextPending();
      if (nextUid && nextUid !== sessionUid) {
        nextLink.href = buildSessionDetailHref(nextUid);
        nextLink.textContent = "Next Pending";
      } else {
        nextLink.href = buildUiHrefWithDate("review-queue.html");
        nextLink.textContent = "No Pending";
      }
    }

    const machine = sopStatusValue(payload, "machine");
    const final = sopStatusValue(payload, "final");
    const review = String(payload.review_status || (payload.review && payload.review.review_status) || "PENDING");
    const machinePill = document.getElementById("detail-machine-status");
    const reviewPill = document.getElementById("detail-review-status");
    const finalPill = document.getElementById("detail-final-status");
    if (overviewAi) {
      overviewAi.textContent = displayStepStatus(machine);
    }
    if (overviewStatusCard instanceof HTMLElement) {
      overviewStatusCard.classList.remove("status-yes", "status-no", "status-unknown");
      const machineKey = String(machine || "").toUpperCase();
      if (machineKey === "DONE") overviewStatusCard.classList.add("status-yes");
      else if (machineKey === "NOT_DONE") overviewStatusCard.classList.add("status-no");
      else overviewStatusCard.classList.add("status-unknown");
    }
    if (machinePill) {
      machinePill.className = `pill ${pillClassForStepStatus(machine)}`;
      machinePill.textContent = `AI ${displayStepStatus(machine)}`;
    }
    if (reviewPill) {
      reviewPill.className = `pill ${pillClassForReviewStatus(review)}`;
      reviewPill.textContent = `Review ${displayReviewStatus(review)}`;
    }
    if (finalPill) {
      finalPill.className = `pill ${pillClassForStepStatus(final)}`;
      finalPill.textContent = `Final ${displayStepStatus(final)}`;
    }

    renderSopPanel(payload);

    const noteBox = document.getElementById("review-note");
    if (noteBox instanceof HTMLTextAreaElement) {
      noteBox.value = payload.review && payload.review.review_note ? String(payload.review.review_note) : "";
    }

    const player = document.querySelector(".detail-player");
    const evidenceStrip = document.getElementById("detail-evidence-strip");
    const evidenceState = document.getElementById("detail-evidence-state");
    const evidencePill = document.getElementById("detail-evidence-pill");
    if (player instanceof HTMLElement) {
      const renderThumbnail = () => {
        if (payload.thumbnail_url) {
          const url = String(payload.thumbnail_url);
          player.innerHTML = `<img alt="thumbnail" src="${url}" style="width:100%;height:100%;object-fit:cover;border-radius:inherit;" />`;
          return;
        }
        player.innerHTML = '<div class="frame-overlay"><span class="pill">tanpa bukti</span></div>';
      };

      const clipsRaw = Array.isArray(payload.clips) ? payload.clips : [];
      const clips = clipsRaw
        .filter((clip) => clip && clip.url)
        .map((clip) => {
          const directUrl = String(clip.url);
          const playbackUrl = clip.playback_url ? String(clip.playback_url) : directUrl;
          return {
            ...clip,
            url: directUrl,
            playback_url: playbackUrl,
          };
        });
      const annotatedUrl = payload.annotated_url ? String(payload.annotated_url) : "";
      const annotatedPlaybackUrl = payload.annotated_playback_url ? String(payload.annotated_playback_url) : annotatedUrl;
      const hasAnnotated = Boolean(annotatedUrl);
      const finalSopStatus = sopStatusValue(payload, "final");
      const preferAnnotated = hasAnnotated && finalSopStatus === "UNKNOWN";

      const setEvidenceState = () => {
        if (!(evidenceState instanceof HTMLElement)) return;
        if (preferAnnotated) {
          evidenceState.className = "pill dir-b";
          evidenceState.textContent = "unknown: video anotasi penuh";
          if (evidencePill) evidencePill.textContent = "Bukti video penuh";
          return;
        }
        if (clips.length > 0) {
          evidenceState.className = "pill brand";
          evidenceState.textContent = `${clips.length} klip tersedia`;
          if (evidencePill) evidencePill.textContent = `${clips.length} klip bukti`;
          return;
        }
        if (hasAnnotated) {
          evidenceState.className = "pill brand";
          evidenceState.textContent = "video anotasi penuh";
          if (evidencePill) evidencePill.textContent = "Bukti video penuh";
          return;
        }
        if (payload.thumbnail_url) {
          evidenceState.className = "pill dir-b";
          evidenceState.textContent = "thumbnail saja";
          if (evidencePill) evidencePill.textContent = "Bukti thumbnail";
          return;
        }
        evidenceState.className = "pill no";
        evidenceState.textContent = "tidak ada bukti";
        if (evidencePill) evidencePill.textContent = "Bukti tidak ada";
      };

      const setActiveClipPill = (activeKey) => {
        if (!(evidenceStrip instanceof HTMLElement)) return;
        const pills = evidenceStrip.querySelectorAll("button[data-clip-key]");
        pills.forEach((pill) => {
          const key = pill.getAttribute("data-clip-key") || "";
          if (key === activeKey) {
            pill.classList.add("active");
          } else {
            pill.classList.remove("active");
          }
        });
      };

      const renderVideoClip = (playbackUrl, directUrl) => {
        player.innerHTML =
          '<video id="detail-video" controls preload="metadata" playsinline style="width:100%;height:100%;object-fit:cover;border-radius:inherit;"></video><div class="frame-overlay"><a id="detail-clip-link" class="pill" target="_blank" rel="noreferrer">buka file klip</a></div>';
        const video = document.getElementById("detail-video");
        const clipLink = document.getElementById("detail-clip-link");
        if (clipLink instanceof HTMLAnchorElement) {
          clipLink.href = directUrl;
        }
        if (video instanceof HTMLVideoElement) {
          video.src = playbackUrl;
          video.addEventListener(
            "error",
            () => {
              // Browser cannot decode this clip codec (common with OpenCV mp4v).
              renderThumbnail();
            },
            { once: true }
          );
        }
      };

      const renderEvidenceStrip = () => {
        if (!(evidenceStrip instanceof HTMLElement)) return;
        evidenceStrip.innerHTML = "";

        if (clips.length > 0) {
          clips.forEach((clip, idx) => {
            const clipNameRaw = clip && clip.name ? String(clip.name) : `clip_${idx + 1}`;
            const clipName = clipNameRaw.replaceAll("_", " ");
            const eventS = Number(clip && clip.event_time_s);
            const timeTag = Number.isFinite(eventS) ? ` +${eventS.toFixed(1)}s` : "";
            const clipKey = `clip-${idx}`;
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "pill detail-clip-pill";
            btn.setAttribute("data-clip-key", clipKey);
            btn.textContent = `${clipName}${timeTag}`;
            btn.addEventListener("click", () => {
              renderVideoClip(String(clip.playback_url || clip.url), String(clip.url));
              setActiveClipPill(clipKey);
            });
            evidenceStrip.appendChild(btn);
          });
        } else {
          const noClipPill = document.createElement("span");
          noClipPill.className = "pill";
          noClipPill.textContent = "tidak ada event klip";
          evidenceStrip.appendChild(noClipPill);
        }

        if (hasAnnotated) {
          const annotatedBtn = document.createElement("button");
          annotatedBtn.type = "button";
          annotatedBtn.className = "pill detail-clip-pill";
          annotatedBtn.setAttribute("data-clip-key", "annotated");
          annotatedBtn.textContent = "video anotasi penuh";
          annotatedBtn.addEventListener("click", () => {
            renderVideoClip(annotatedPlaybackUrl || annotatedUrl, annotatedUrl);
            setActiveClipPill("annotated");
          });
          evidenceStrip.appendChild(annotatedBtn);
        }

      };

      setEvidenceState();
      renderEvidenceStrip();
      if (preferAnnotated) {
        renderVideoClip(annotatedPlaybackUrl || annotatedUrl, annotatedUrl);
        setActiveClipPill("annotated");
      } else if (clips.length > 0) {
        renderVideoClip(String(clips[0].playback_url || clips[0].url), String(clips[0].url));
        setActiveClipPill("clip-0");
      } else if (hasAnnotated) {
        renderVideoClip(annotatedPlaybackUrl || annotatedUrl, annotatedUrl);
        setActiveClipPill("annotated");
      } else {
        renderThumbnail();
      }
    }

    if (form) {
      form.onsubmit = async (event) => {
        event.preventDefault();
        const reviewStatus = statusInput instanceof HTMLInputElement ? statusInput.value : "PENDING";
        const note = noteBox instanceof HTMLTextAreaElement ? noteBox.value : "";
        const overrides = {};
        form.querySelectorAll("select[data-override-key]").forEach((node) => {
          if (!(node instanceof HTMLSelectElement)) return;
          const key = node.getAttribute("data-override-key") || "";
          const value = String(node.value || "").trim();
          if (key && value) overrides[key] = value;
        });
        const body = { review_status: reviewStatus, review_note: note };
        if (Object.keys(overrides).length > 0) {
          body.overrides = overrides;
        }
        try {
          await apiFetchJson(`/api/sessions/${encodeURIComponent(sessionUid)}/review`, {
            method: "PUT",
            body: JSON.stringify(body),
          });
          if (reviewPill) {
            reviewPill.className = `pill ${pillClassForReviewStatus(reviewStatus)}`;
            reviewPill.textContent = `Review ${displayReviewStatus(reviewStatus)}`;
          }
          if (String(reviewStatus || "").toUpperCase() !== "PENDING") {
            const nextUid = await resolveNextPending();
            if (nextUid && nextUid !== sessionUid) {
              window.location.assign(buildSessionDetailHref(nextUid));
              return;
            }
            window.location.assign(buildUiHrefWithDate("review-queue.html"));
          }
        } catch (err) {
          alert("Failed to save review");
        }
      };
    }
  };

  const populateSetup = async () => {
    const dataDir = document.getElementById("admin-data-dir");
    const dbPath = document.getElementById("admin-db-path");
    const lastScan = document.getElementById("admin-last-scan");
    const sessionCount = document.getElementById("admin-session-count");
    const rescanBtn = document.getElementById("admin-rescan");
    const autoApproveEnabled = document.getElementById("admin-auto-approve-enabled");
    const autoApproveMinDuration = document.getElementById("admin-auto-approve-min-duration");

    const diskFree = document.getElementById("admin-disk-free");
    const diskTotal = document.getElementById("admin-disk-total");
    const clipCount = document.getElementById("admin-clip-count");
    const thumbCount = document.getElementById("admin-thumb-count");
    const annotatedCount = document.getElementById("admin-annotated-count");
    const storageRefreshBtn = document.getElementById("admin-storage-refresh");
    const storageTestBtn = document.getElementById("admin-storage-test");
    const storageTestStatus = document.getElementById("admin-storage-test-status");
    const opsRefreshBtn = document.getElementById("admin-ops-refresh");

    const opsServiceState = document.getElementById("admin-ops-service-state");
    const opsServicePill = document.getElementById("admin-ops-service-pill");
    const opsManagedTotal = document.getElementById("admin-ops-managed-total");
    const opsManagedMeta = document.getElementById("admin-ops-managed-meta");
    const opsVideoTotal = document.getElementById("admin-ops-video-total");
    const opsVideoMeta = document.getElementById("admin-ops-video-meta");
    const opsDiskFree = document.getElementById("admin-ops-disk-free");
    const opsDiskUsed = document.getElementById("admin-ops-disk-used");
    const opsSpoolRoot = document.getElementById("admin-ops-spool-root");
    const opsPendingFiles = document.getElementById("admin-ops-pending-files");
    const opsDoneFiles = document.getElementById("admin-ops-done-files");
    const opsDeadFiles = document.getElementById("admin-ops-dead-files");
    const opsSpoolHealth = document.getElementById("admin-ops-spool-health");
    const opsPendingNewest = document.getElementById("admin-ops-pending-newest");
    const opsDbPill = document.getElementById("admin-ops-db-pill");
    const opsDbSize = document.getElementById("admin-ops-db-size");
    const opsEvidenceSize = document.getElementById("admin-ops-evidence-size");
    const opsAnnotatedSize = document.getElementById("admin-ops-annotated-size");
    const opsSessionMetaSize = document.getElementById("admin-ops-session-meta-size");
    const opsPlatformSize = document.getElementById("admin-ops-platform-size");
    const opsReportsFiles = document.getElementById("admin-ops-reports-files");
    const opsCacheSize = document.getElementById("admin-ops-cache-size");
    const opsCacheUpdated = document.getElementById("admin-ops-cache-updated");
    const opsActionPill = document.getElementById("admin-ops-action-pill");
    const opsCallout = document.getElementById("admin-ops-callout");
    const opsRetentionHint = document.getElementById("admin-ops-retention-hint");
    const opsBackupHint = document.getElementById("admin-ops-backup-hint");

    const formatBytes = (bytes) => {
      const b = Number(bytes || 0);
      if (!Number.isFinite(b) || b <= 0) return "-";
      const units = ["B", "KB", "MB", "GB", "TB"];
      let v = b;
      let u = 0;
      while (v >= 1024 && u < units.length - 1) {
        v /= 1024;
        u += 1;
      }
      return `${v.toFixed(u === 0 ? 0 : 1)} ${units[u]}`;
    };

    const formatUtc = (raw) => {
      if (!raw) return "-";
      const d = new Date(raw);
      if (Number.isNaN(d.getTime())) return String(raw);
      return (
        d.toLocaleString("en-GB", {
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
          timeZone: "UTC",
        }) + " UTC"
      );
    };

    const setPill = (node, label, tone) => {
      if (!node) return;
      node.className = `pill ${tone || ""}`.trim();
      node.textContent = label;
    };

    const refresh = async () => {
      try {
        const cfg = await apiFetchJson("/api/config");
        if (dataDir) dataDir.textContent = String(cfg.data_dir || "-");
        if (dbPath) dbPath.textContent = String(cfg.db_path || "-");
        if (lastScan) lastScan.textContent = String(cfg.last_scan_utc || "-");
        if (sessionCount) sessionCount.textContent = String(cfg.session_count ?? "-");
        if (autoApproveEnabled) {
          const on = Boolean(cfg.auto_approve_done_enabled);
          autoApproveEnabled.className = `pill ${on ? "yes" : "no"}`;
          autoApproveEnabled.textContent = on ? "enabled" : "disabled";
        }
        if (autoApproveMinDuration) {
          const v = Number(cfg.auto_approve_min_duration_s);
          autoApproveMinDuration.textContent = Number.isFinite(v) ? `${v.toFixed(1)} s` : "-";
        }
      } catch (err) {
        if (dataDir) dataDir.textContent = "Failed to load";
      }
    };

    const refreshStorage = async () => {
      try {
        const s = await apiFetchJson("/api/admin/storage");
        if (diskFree) diskFree.textContent = formatBytes(s.disk_free_bytes);
        if (diskTotal) diskTotal.textContent = formatBytes(s.disk_total_bytes);
        if (clipCount) clipCount.textContent = String(s.clip_count ?? "-");
        if (thumbCount) thumbCount.textContent = String(s.thumbnail_count ?? "-");
        if (annotatedCount) annotatedCount.textContent = String(s.annotated_count ?? "-");
      } catch (err) {
        if (diskFree) diskFree.textContent = "Failed to load";
      }
    };

    const refreshOps = async () => {
      const payload = await apiFetchJson("/api/admin/ops");
      const disk = payload && payload.disk ? payload.disk : {};
      const db = payload && payload.database ? payload.database : {};
      const reports = payload && payload.reports ? payload.reports : {};
      const cache = payload && payload.cache ? payload.cache : {};
      const spool = payload && payload.uploader_spool ? payload.uploader_spool : {};
      const spoolHealth = spool && spool.health ? spool.health : {};
      const spoolStateFile = spool && spool.state_file ? spool.state_file : {};
      const managed = payload && payload.managed_storage ? payload.managed_storage : {};
      const managedSessions = managed && managed.sessions ? managed.sessions : {};
      const categories = managedSessions && managedSessions.categories ? managedSessions.categories : {};
      const pending = spool && spool.pending ? spool.pending : {};
      const done = spool && spool.done ? spool.done : {};
      const dead = spool && spool.dead ? spool.dead : {};

      const freeBytes = Number(disk.free_bytes || 0);
      const totalBytes = Number(disk.total_bytes || 0);
      const usedBytes = Number(disk.used_bytes || 0);
      const pendingFiles = Number(pending.files || 0);
      const deadFiles = Number(dead.files || 0);
      const managedTotalBytes = Number(managed.total_bytes || 0);
      const evidenceClipBytes = Number(categories.evidence_clips && categories.evidence_clips.bytes ? categories.evidence_clips.bytes : 0);
      const evidenceClipFiles = Number(categories.evidence_clips && categories.evidence_clips.files ? categories.evidence_clips.files : 0);
      const annotatedBytes = Number(categories.annotated_videos && categories.annotated_videos.bytes ? categories.annotated_videos.bytes : 0);
      const annotatedFiles = Number(categories.annotated_videos && categories.annotated_videos.files ? categories.annotated_videos.files : 0);
      const checklistBytes = Number(categories.checklists && categories.checklists.bytes ? categories.checklists.bytes : 0);
      const runConfigBytes = Number(categories.run_configs && categories.run_configs.bytes ? categories.run_configs.bytes : 0);
      const thumbnailBytes = Number(categories.thumbnails && categories.thumbnails.bytes ? categories.thumbnails.bytes : 0);
      const evidenceManifestBytes = Number(categories.evidence_manifests && categories.evidence_manifests.bytes ? categories.evidence_manifests.bytes : 0);
      const sessionMetaBytes = checklistBytes + runConfigBytes + thumbnailBytes + evidenceManifestBytes;
      const platformBytes =
        Number(reports.bytes || 0) +
        Number(cache.bytes || 0) +
        Number(managed.database && managed.database.bytes ? managed.database.bytes : 0) +
        Number(managed.uploader_spool && managed.uploader_spool.bytes ? managed.uploader_spool.bytes : 0);
      const videoBytes = evidenceClipBytes + annotatedBytes;

      if (opsManagedTotal) opsManagedTotal.textContent = formatBytes(managedTotalBytes);
      if (opsManagedMeta) {
        opsManagedMeta.textContent = `${String(managed.total_files || 0)} managed files across sessions, reports, cache, DB, and spool`;
      }
      if (opsVideoTotal) opsVideoTotal.textContent = formatBytes(videoBytes);
      if (opsVideoMeta) opsVideoMeta.textContent = `${evidenceClipFiles} evidence clips · ${annotatedFiles} annotated video(s)`;
      if (opsDiskFree) opsDiskFree.textContent = formatBytes(freeBytes);
      if (opsDiskUsed) {
        const usedPct = totalBytes > 0 ? (usedBytes * 100.0) / totalBytes : 0;
        opsDiskUsed.textContent = `used ${formatBytes(usedBytes)} (${usedPct.toFixed(1)}%)`;
      }
      if (opsSpoolRoot) setPill(opsSpoolRoot, spool.exists ? "spool online" : "spool missing", spool.exists ? "ink" : "no");
      if (opsPendingFiles) {
        const pendingHint = pending.oldest_item_utc ? `oldest ${formatUtc(pending.oldest_item_utc)}` : "queue empty";
        opsPendingFiles.textContent = `${String(pendingFiles)} · ${formatBytes(pending.bytes)} · ${pendingHint}`;
      }
      if (opsDoneFiles) opsDoneFiles.textContent = `${String(done.files || 0)} · ${formatBytes(done.bytes)}`;
      if (opsDeadFiles) {
        const deadHint = dead.oldest_item_utc ? `oldest ${formatUtc(dead.oldest_item_utc)}` : "no dead letters";
        opsDeadFiles.textContent = `${String(deadFiles)} · ${formatBytes(dead.bytes)} · ${deadHint}`;
      }
      if (opsSpoolHealth) {
        const healthStatus = String(spoolHealth.status || "unknown").toLowerCase();
        const healthTone = healthStatus === "ok" ? "yes" : healthStatus === "warning" ? "pending" : healthStatus === "error" ? "no" : "";
        const issueCount = Array.isArray(spoolHealth.issues) ? spoolHealth.issues.length : 0;
        const healthLabel = issueCount > 0 ? `${healthStatus} (${issueCount})` : healthStatus;
        setPill(opsSpoolHealth, healthLabel, healthTone);
      }
      if (opsPendingNewest) opsPendingNewest.textContent = pending.newest_item_utc ? formatUtc(pending.newest_item_utc) : "-";
      if (opsDbSize) opsDbSize.textContent = db.exists ? `db ${formatBytes(db.bytes)}` : "db missing";
      if (opsDbPill) setPill(opsDbPill, db.exists ? "db online" : "db missing", db.exists ? "brand" : "no");
      if (opsEvidenceSize) opsEvidenceSize.textContent = `${formatBytes(evidenceClipBytes)} · ${evidenceClipFiles} clip(s)`;
      if (opsAnnotatedSize) opsAnnotatedSize.textContent = `${formatBytes(annotatedBytes)} · ${annotatedFiles} file(s)`;
      if (opsSessionMetaSize) opsSessionMetaSize.textContent = formatBytes(sessionMetaBytes);
      if (opsPlatformSize) opsPlatformSize.textContent = formatBytes(platformBytes);
      if (opsReportsFiles) opsReportsFiles.textContent = `reports ${String(reports.files || 0)} · ${formatBytes(reports.bytes)}`;
      if (opsCacheSize) opsCacheSize.textContent = `cache ${String(cache.files || 0)} · ${formatBytes(cache.bytes)}`;
      if (opsCacheUpdated) opsCacheUpdated.textContent = cache.last_modified_utc ? `cache updated ${formatUtc(cache.last_modified_utc)}` : "cache updated -";

      let stateLabel = "Stable";
      let stateTone = "yes";
      let actionLabel = "monitor";
      let callout =
        "System looks healthy. Keep uploader watch mode running and use retention dry-run before cleanup.";
      if (!db.exists || !spool.exists) {
        stateLabel = "Attention";
        stateTone = "no";
        actionLabel = "repair";
        callout = "Core storage paths are missing. Verify `data/`, SQLite DB path, and uploader spool initialization first.";
      } else if (deadFiles > 0) {
        stateLabel = "Dead letters";
        stateTone = "no";
        actionLabel = "recover";
        callout = "Uploader has dead tasks. Inspect `data/uploader_spool/dead/`, fix the cause, then rerun uploader one-shot or watch mode.";
      } else if (spoolStateFile.is_stale === true) {
        stateLabel = "Stale watch";
        stateTone = "pending";
        actionLabel = "check";
        callout = "Uploader spool heartbeat looks stale. Confirm watch mode is still running and that `state.json` is updating.";
      } else if (pendingFiles > 0) {
        stateLabel = "Sync backlog";
        stateTone = "pending";
        actionLabel = "observe";
        callout = "Uploads are queued but not failed. If this count keeps growing, check website availability and uploader watch mode.";
      } else if (freeBytes > 0 && totalBytes > 0 && freeBytes / totalBytes < 0.15) {
        stateLabel = "Low disk";
        stateTone = "pending";
        actionLabel = "cleanup";
        callout = "Free disk is below 15%. Run retention dry-run, archive old sessions, and confirm backup before deleting media.";
      }

      if (opsServiceState) opsServiceState.textContent = stateLabel;
      if (opsServicePill) setPill(opsServicePill, stateLabel.toLowerCase(), stateTone);
      if (opsActionPill) setPill(opsActionPill, actionLabel, stateTone === "yes" ? "ink" : stateTone);
      if (opsCallout) opsCallout.textContent = callout;
      if (opsRetentionHint) {
        opsRetentionHint.textContent =
          freeBytes > 0 && totalBytes > 0 && freeBytes / totalBytes < 0.15
            ? "Disk pressure is visible. Run cleanup dry-run soon, then apply only after backup."
            : "Retention is healthy. Use cleanup dry-run weekly to control cache, evidence clips, and old uploader tasks.";
      }
      if (opsBackupHint) {
        opsBackupHint.textContent = db.last_modified_utc
          ? `Last DB write seen ${formatUtc(db.last_modified_utc)}. Back up videos, reports, and SQLite together.`
          : "Back up videos, reports, and SQLite together before maintenance or manual cleanup.";
      }
    };

    await refresh();
    await refreshStorage();
    try {
      await refreshOps();
    } catch (err) {
      if (opsServiceState) opsServiceState.textContent = "Failed";
      if (opsServicePill) setPill(opsServicePill, "unavailable", "no");
      if (opsCallout) opsCallout.textContent = "Failed to load /api/admin/ops. Check auth, server logs, or FastAPI route wiring.";
    }

    if (rescanBtn instanceof HTMLButtonElement) {
      rescanBtn.addEventListener("click", async () => {
        rescanBtn.setAttribute("disabled", "disabled");
        try {
          const res = await apiFetchJson("/api/admin/rescan", { method: "POST" });
          if (lastScan) lastScan.textContent = String(res.last_scan_utc || "-");
          if (sessionCount) sessionCount.textContent = String(res.session_count ?? "-");
          await refreshStorage();
          await refreshOps();
        } catch (err) {
          alert("Rescan gagal");
        } finally {
          rescanBtn.removeAttribute("disabled");
        }
      });
    }

    if (storageRefreshBtn instanceof HTMLButtonElement) {
      storageRefreshBtn.addEventListener("click", async () => {
        storageRefreshBtn.setAttribute("disabled", "disabled");
        try {
          await refreshStorage();
          await refreshOps();
        } finally {
          storageRefreshBtn.removeAttribute("disabled");
        }
      });
    }

    if (opsRefreshBtn instanceof HTMLButtonElement) {
      opsRefreshBtn.addEventListener("click", async () => {
        opsRefreshBtn.setAttribute("disabled", "disabled");
        try {
          await refresh();
          await refreshStorage();
          await refreshOps();
        } catch (err) {
          alert("Ops summary gagal dimuat");
        } finally {
          opsRefreshBtn.removeAttribute("disabled");
        }
      });
    }

    if (storageTestBtn instanceof HTMLButtonElement) {
      storageTestBtn.addEventListener("click", async () => {
        storageTestBtn.setAttribute("disabled", "disabled");
        if (storageTestStatus) {
          storageTestStatus.className = "caption";
          storageTestStatus.textContent = "Testing storage write access...";
        }
        try {
          await apiFetchJson("/api/admin/storage/test", { method: "POST" });
          if (storageTestStatus) {
            storageTestStatus.className = "caption";
            storageTestStatus.textContent = "Storage test: OK.";
          }
          await refreshOps();
        } catch (err) {
          if (storageTestStatus) {
            storageTestStatus.className = "caption";
            storageTestStatus.textContent = "Storage test failed. Check server logs and data directory permissions.";
          }
        } finally {
          storageTestBtn.removeAttribute("disabled");
        }
      });
    }
  };

  const bindStaleIndicator = ({ pillId, onRefresh }) => {
    const pill = document.getElementById(pillId);
    if (!(pill instanceof HTMLElement)) return;
    if (typeof onRefresh !== "function") return;

    let baselineLastScan = "";
    let baselineCount = null;
    let lastSeenCfg = null;
    let running = false;

    const loadCfg = async () => {
      try {
        const cfg = await apiFetchJson("/api/config");
        return cfg && typeof cfg === "object" ? cfg : null;
      } catch (err) {
        return null;
      }
    };

    const updateBaseline = (cfg) => {
      baselineLastScan = cfg && cfg.last_scan_utc ? String(cfg.last_scan_utc) : "";
      const n = cfg && cfg.session_count != null ? Number(cfg.session_count) : Number.NaN;
      baselineCount = Number.isFinite(n) ? n : null;
    };

    const setVisible = (show, text) => {
      if (show) {
        if (text) pill.textContent = String(text);
        pill.removeAttribute("hidden");
      } else {
        pill.setAttribute("hidden", "hidden");
      }
    };

    const pollOnce = async () => {
      if (running) return;
      if (document.visibilityState === "hidden") return;
      running = true;
      try {
        const cfg = await loadCfg();
        if (!cfg) return;
        lastSeenCfg = cfg;

        const nextLast = cfg.last_scan_utc ? String(cfg.last_scan_utc) : "";
        const nextCount = cfg.session_count != null ? Number(cfg.session_count) : Number.NaN;
        const countChanged = Number.isFinite(nextCount) && baselineCount != null && nextCount !== baselineCount;
        const scanChanged = Boolean(nextLast && baselineLastScan && nextLast !== baselineLastScan);

        if (countChanged || scanChanged) {
          setVisible(true, "New data available (click to refresh)");
        }
      } finally {
        running = false;
      }
    };

    pill.addEventListener("click", async () => {
      pill.setAttribute("aria-busy", "true");
      try {
        await Promise.resolve(onRefresh());
        if (lastSeenCfg) updateBaseline(lastSeenCfg);
        setVisible(false);
      } finally {
        pill.removeAttribute("aria-busy");
      }
    });

    // Initialize baseline and start polling.
    (async () => {
      const cfg = await loadCfg();
      if (cfg) updateBaseline(cfg);
      setVisible(false);
      pollOnce();
      window.setInterval(pollOnce, 15000);
      document.addEventListener("visibilitychange", () => {
        pollOnce();
      });
    })();
  };

  const populateDashboard = async () => {
    const totalNode = document.getElementById("kpi-total-sessions");
    const pendingNode = document.getElementById("kpi-pending");
    const dashboardBody = document.body;
    if (!totalNode && !pendingNode) {
      return;
    }
    if (dashboardBody && dashboardBody.classList.contains("page-dashboard")) {
      dashboardBody.classList.add("is-hydrating");
    }
    const dashboardDateLabel = document.getElementById("dashboard-active-date-slice");
    if (dashboardDateLabel) dashboardDateLabel.textContent = dateSliceLabel();
    try {
      const s = await apiFetchJson(withDateApiQuery("/api/stats"));
      if (totalNode) totalNode.textContent = String(s.total_sessions ?? "-");
      if (pendingNode) pendingNode.textContent = String(s.pending ?? "-");

      const totalHint = document.getElementById("kpi-total-hint");
      if (totalHint) {
        totalHint.textContent = `lolos ${String(s.approved ?? 0)} | tidak lolos ${String(s.rejected ?? 0)}`;
      }

      const pendingHint = document.getElementById("kpi-pending-hint");
      if (pendingHint) {
        const reviewed = Number(s.reviewed ?? 0);
        const total = Number(s.total_sessions ?? 0);
        const completionPct = Number(s.review_completion_pct ?? 0);
        pendingHint.textContent =
          total > 0
            ? `${reviewed}/${total} decided (${completionPct.toFixed(1)}%)`
            : "Belum ada sesi";
      }

      const bannerPending = document.getElementById("banner-pending-pill");
      if (bannerPending) bannerPending.textContent = `pending ${String(s.pending ?? "-")}`;

      const helmetCheckedNode = document.getElementById("kpi-helmet-checked");
      if (helmetCheckedNode) helmetCheckedNode.textContent = String(s.final_sop_done ?? "-");
      const helmetCheckedHint = document.getElementById("kpi-helmet-checked-hint");
      if (helmetCheckedHint) {
        const pct = Number(s.reviewed_final_sop_done_pct ?? 0);
        helmetCheckedHint.textContent = `${pct.toFixed(1)}% DONE across reviewed sessions`;
      }

      const unknownRateNode = document.getElementById("kpi-unknown-rate");
      if (unknownRateNode) {
        const pct = Number(s.final_sop_unknown_pct ?? 0);
        unknownRateNode.textContent = `${pct.toFixed(1)}%`;
      }
      const unknownRateHint = document.getElementById("kpi-unknown-rate-hint");
      if (unknownRateHint) {
        unknownRateHint.textContent = `${String(s.final_sop_unknown ?? 0)} UNKNOWN from final SOP status`;
      }

      const manualReviewNode = document.getElementById("kpi-manual-review");
      if (manualReviewNode) manualReviewNode.textContent = String(s.human_reviewed ?? s.reviewed ?? "-");
      const manualReviewHint = document.getElementById("kpi-manual-review-hint");
      if (manualReviewHint) {
        manualReviewHint.textContent = `${String(s.manual_overrides ?? 0)} manual overrides`;
      }

      const compactManualNeeded = document.getElementById("dashboard-compact-manual-needed");
      if (compactManualNeeded) compactManualNeeded.textContent = String(s.final_sop_unknown ?? "-");
      const compactApproved = document.getElementById("dashboard-compact-approved");
      if (compactApproved) compactApproved.textContent = String(s.approved ?? "-");
      const compactMachineNo = document.getElementById("dashboard-compact-machine-no");
      if (compactMachineNo) compactMachineNo.textContent = String(s.machine_sop_not_done ?? "-");
      const compactPending = document.getElementById("dashboard-compact-pending");
      if (compactPending) compactPending.textContent = String(s.pending ?? "-");

      const trendDone = document.getElementById("trend-strip-done");
      if (trendDone) {
        trendDone.textContent = `DONE ${String(s.machine_sop_done ?? 0)}`;
      }
      const trendUnknown = document.getElementById("trend-strip-unknown");
      if (trendUnknown) {
        const unknownPct = Number(s.final_sop_unknown_pct ?? 0);
        trendUnknown.textContent = `UNKNOWN ${String(s.machine_sop_unknown ?? 0)} (${unknownPct.toFixed(1)}%)`;
      }
      const trendNotDone = document.getElementById("trend-strip-not-done");
      if (trendNotDone) trendNotDone.textContent = `NOT DONE ${String(s.machine_sop_not_done ?? 0)}`;

      const renderDashboardTrend = async () => {
        const svg = document.querySelector(".trend-svg");
        if (!(svg instanceof SVGSVGElement)) return;

        const trendSlice = document.getElementById("trend-active-slice");
        const trendXTitle = document.getElementById("trend-x-title");

        const donePath = svg.querySelector("path.trend-line.dir-a");
        const unknownPath = svg.querySelector("path.trend-line.dir-b");
        const notDonePath = svg.querySelector("path.trend-line.no");
        const doneDot = svg.querySelector("circle.trend-focus-dot.dir-a");
        const unknownDot = svg.querySelector("circle.trend-focus-dot.dir-b");
        const notDoneDot = svg.querySelector("circle.trend-focus-dot.no");
        const gridValues = Array.from(svg.querySelectorAll("text.trend-grid-value"));
        const axisLabels = Array.from(svg.querySelectorAll("text.trend-axis-label"));

        if (
          !(donePath instanceof SVGPathElement) ||
          !(unknownPath instanceof SVGPathElement) ||
          !(notDonePath instanceof SVGPathElement)
        ) {
          return;
        }

        const slice = readDateSliceFromUrl();

        let listPayload;
        let listUrl = "";
        try {
          if (slice.from || slice.to) {
            listUrl = withDateApiQuery("/api/sessions?limit=2000&sort=OLDEST");
          } else {
            let activeDate = null;
            try {
              const latest = await apiFetchJson("/api/sessions?limit=1&sort=NEWEST");
              const sessions = Array.isArray(latest.sessions) ? latest.sessions : [];
              if (sessions.length > 0 && sessions[0] && sessions[0].date) {
                activeDate = String(sessions[0].date);
              }
            } catch (err) {
              // ignore
            }
            listUrl = activeDate
              ? `/api/sessions?limit=2000&sort=OLDEST&date=${encodeURIComponent(activeDate)}`
              : "/api/sessions?limit=2000&sort=OLDEST";
          }
          listPayload = await apiFetchJson(listUrl);
        } catch (err) {
          return;
        }

        const sessions = Array.isArray(listPayload.sessions) ? listPayload.sessions : [];

        const useHourly =
          (slice.from && slice.to && slice.from === slice.to) ||
          (!slice.from && !slice.to && typeof listUrl === "string" && listUrl.includes("date="));

        if (trendXTitle instanceof HTMLElement) {
          trendXTitle.textContent = useHourly ? "Time (local)" : "Date";
        }

        if (trendSlice instanceof HTMLElement) {
          if (useHourly) {
            let activeDate = "";
            if (slice.from && slice.to && slice.from === slice.to) {
              activeDate = slice.from;
            } else {
              try {
                const u = new URL(listUrl, window.location.href);
                activeDate = String(u.searchParams.get("date") || "");
              } catch (err) {
                // ignore
              }
            }
            if (!activeDate && sessions.length > 0 && sessions[0] && sessions[0].date) {
              activeDate = String(sessions[0].date || "");
            }
            trendSlice.textContent = activeDate ? `Date: ${activeDate}` : "Date: (auto)";
          } else {
            // Keep this aligned with the rest of the dashboard date filter.
            trendSlice.textContent = dateSliceLabel();
          }
        }

        let labels = [];
        let done = [];
        let unknown = [];
        let notDone = [];

        if (useHourly) {
          labels = Array.from({ length: 24 }, (_, h) => `${String(h).padStart(2, "0")}:00`);
          done = Array(24).fill(0);
          unknown = Array(24).fill(0);
          notDone = Array(24).fill(0);
          sessions.forEach((row) => {
            const iso =
              row && row.start_time_iso ? String(row.start_time_iso) : row && row.end_time_iso ? String(row.end_time_iso) : "";
            if (!iso) return;
            const dt = new Date(iso);
            if (Number.isNaN(dt.getTime())) return;
            const idx = dt.getHours();
            if (idx < 0 || idx >= 24) return;
            const status = sopStatusValue(row, "machine");
            if (status === "DONE") done[idx] += 1;
            else if (status === "NOT_DONE") notDone[idx] += 1;
            else unknown[idx] += 1;
          });
        } else {
          // Daily buckets across the filtered sessions window.
          const ymdRe = /^\d{4}-\d{2}-\d{2}$/;

          if (slice.from && slice.to && ymdRe.test(slice.from) && ymdRe.test(slice.to) && slice.from <= slice.to) {
            const out = [];
            const start = new Date(`${slice.from}T00:00:00`);
            const end = new Date(`${slice.to}T00:00:00`);
            if (!Number.isNaN(start.getTime()) && !Number.isNaN(end.getTime())) {
              const d = new Date(start.getTime());
              while (d.getTime() <= end.getTime() && out.length < 370) {
                out.push(toYmdLocal(d));
                d.setDate(d.getDate() + 1);
              }
            }
            labels = out.length ? out : [slice.from];
          } else {
            const dates = Array.from(
              new Set(
                sessions
                  .map((row) => (row && row.date ? String(row.date) : ""))
                  .filter((v) => v && ymdRe.test(v))
              )
            ).sort();

            labels = dates.length ? dates : [slice.from || slice.to || "-"];
          }
          done = Array(labels.length).fill(0);
          unknown = Array(labels.length).fill(0);
          notDone = Array(labels.length).fill(0);
          const idxByDate = new Map(labels.map((d, i) => [d, i]));

          sessions.forEach((row) => {
            const d = row && row.date ? String(row.date) : "";
            const idx = idxByDate.get(d);
            if (idx == null) return;
            const status = sopStatusValue(row, "machine");
            if (status === "DONE") done[idx] += 1;
            else if (status === "NOT_DONE") notDone[idx] += 1;
            else unknown[idx] += 1;
          });
        }

        const maxVal = Math.max(1, ...done, ...unknown, ...notDone);
        const view = svg.viewBox && svg.viewBox.baseVal ? svg.viewBox.baseVal : { x: 0, y: 0, width: 760, height: 190 };
        const xMin = 60;
        const xMax = 720;
        const yTop = 26;
        const yBottom = 152;
        const binCount = labels.length;
        const xStep = binCount > 1 ? (xMax - xMin) / (binCount - 1) : 0;
        const yScale = (v) => yBottom - (Math.max(0, Number(v || 0)) / maxVal) * (yBottom - yTop);

        const ptsDone = done.map((v, i) => ({ x: xMin + i * xStep, y: yScale(v) }));
        const ptsUnknown = unknown.map((v, i) => ({ x: xMin + i * xStep, y: yScale(v) }));
        const ptsNotDone = notDone.map((v, i) => ({ x: xMin + i * xStep, y: yScale(v) }));

        const doneD = buildSmoothPath(ptsDone);
        const unknownD = buildSmoothPath(ptsUnknown);
        const notDoneD = buildSmoothPath(ptsNotDone);
        donePath.setAttribute("d", doneD || `M${xMin} ${yBottom} L${xMax} ${yBottom}`);
        unknownPath.setAttribute("d", unknownD || `M${xMin} ${yBottom} L${xMax} ${yBottom}`);
        notDonePath.setAttribute("d", notDoneD || `M${xMin} ${yBottom} L${xMax} ${yBottom}`);

        const lastDone = ptsDone[ptsDone.length - 1];
        const lastUnknown = ptsUnknown[ptsUnknown.length - 1];
        const lastNotDone = ptsNotDone[ptsNotDone.length - 1];
        if (doneDot instanceof SVGCircleElement) {
          doneDot.setAttribute("cx", String(lastDone.x.toFixed(1)));
          doneDot.setAttribute("cy", String(lastDone.y.toFixed(1)));
        }
        if (unknownDot instanceof SVGCircleElement) {
          unknownDot.setAttribute("cx", String(lastUnknown.x.toFixed(1)));
          unknownDot.setAttribute("cy", String(lastUnknown.y.toFixed(1)));
        }
        if (notDoneDot instanceof SVGCircleElement) {
          notDoneDot.setAttribute("cx", String(lastNotDone.x.toFixed(1)));
          notDoneDot.setAttribute("cy", String(lastNotDone.y.toFixed(1)));
        }

        // Update y-axis grid labels to match current scale.
        if (gridValues.length >= 3) {
          const top = maxVal;
          const mid = Math.max(1, Math.round((maxVal * 2) / 3));
          const low = Math.max(1, Math.round(maxVal / 3));
          gridValues[0].textContent = String(top);
          gridValues[1].textContent = String(mid);
          gridValues[2].textContent = String(low);
        }

        const peak = (arr) => {
          let bestVal = -1;
          let bestIdx = 0;
          arr.forEach((v, i) => {
            if (v > bestVal) {
              bestVal = v;
              bestIdx = i;
            }
          });
          return { val: bestVal, idx: bestIdx };
        };

        const donePeak = peak(done);
        const unknownPeak = peak(unknown);
        const notDonePeak = peak(notDone);

        const trendDone = document.getElementById("trend-strip-done");
        if (trendDone) trendDone.textContent = `peak DONE ${Math.max(0, donePeak.val)} @ ${labels[donePeak.idx] || "-"}`;
        const trendUnknown = document.getElementById("trend-strip-unknown");
        if (trendUnknown) trendUnknown.textContent = `peak UNKNOWN ${Math.max(0, unknownPeak.val)} @ ${labels[unknownPeak.idx] || "-"}`;
        const trendNotDone = document.getElementById("trend-strip-not-done");
        if (trendNotDone) trendNotDone.textContent = `peak NOT DONE ${Math.max(0, notDonePeak.val)} @ ${labels[notDonePeak.idx] || "-"}`;

        if (axisLabels.length >= 5 && labels.length >= 2) {
          if (useHourly) {
            const idxs = [0, 6, 12, 18, 23];
            idxs.forEach((idx, i) => {
              const node = axisLabels[i];
              if (node) node.textContent = labels[Math.min(labels.length - 1, Math.max(0, idx))] || "";
            });
          } else {
            const n = labels.length;
            const idxs = [0, Math.floor((n - 1) * 0.25), Math.floor((n - 1) * 0.5), Math.floor((n - 1) * 0.75), n - 1];
            idxs.forEach((idx, i) => {
              const node = axisLabels[i];
              if (!node) return;
              const raw = labels[idx] || "";
              // Shorten YYYY-MM-DD to MM-DD for readability.
              node.textContent = /^\d{4}-\d{2}-\d{2}$/.test(raw) ? raw.slice(5) : raw;
            });
          }
        }
      };

      await renderDashboardTrend();

      const latestBody = document.getElementById("dashboard-latest-sessions-body");
      if (latestBody instanceof HTMLTableSectionElement) {
        try {
          const latestPayload = await apiFetchJson(withDateApiQuery("/api/sessions?limit=3"));
          const latestSessions = Array.isArray(latestPayload.sessions) ? latestPayload.sessions : [];
          if (latestSessions.length === 0) {
            latestBody.innerHTML = `<tr><td colspan="6"><span class="pill">No sessions found</span></td></tr>`;
          } else {
            latestBody.innerHTML = latestSessions
              .map((session) => {
                const uid = String(session.session_uid || "");
                const sid = String(session.session_id || uid || "-");
                const start = formatHmsFromIso(session.start_time_iso);
                const review = String(session.review_status || "PENDING");
                const remark =
                  Number(session.clip_count || 0) > 0
                    ? `${String(session.clip_count)} clip(s) attached`
                    : session.has_thumbnail
                    ? "Thumbnail available"
                    : "No evidence attached";
                return `
                  <tr>
                    <td> <strong>${sid}</strong></td>
                    <td>${start}</td>
                    <td>${dashboardSopPills(session)}</td>
                    <td><span class="pill ${pillClassForReviewStatus(review)}">${displayReviewStatus(review)}</span></td>
                    <td>${remark}</td>
                    <td><a class="btn btn-compact action-inspect" href="${buildSessionDetailHref(uid)}">Inspect</a></td>
                  </tr>
                `;
              })
              .join("");
          }
        } catch (err) {
          latestBody.innerHTML = `<tr><td colspan="6"><span class="pill no">Failed to load sessions</span></td></tr>`;
        }
      }
    } catch (err) {
      // ignore
    } finally {
      if (dashboardBody && dashboardBody.classList.contains("page-dashboard")) {
        dashboardBody.classList.remove("is-hydrating");
      }
    }
  };

  const body = document.body;
  applyDateSliceToStaticNav();
  if (body && body.classList.contains("page-review-queue")) {
    bindStaleIndicator({ pillId: "stale-pill-queue", onRefresh: () => populateQueue() });
    bindDateControls({
      fromId: "queue-date-from",
      toId: "queue-date-to",
      applyId: "queue-date-apply",
      clearId: "queue-date-clear",
      labelId: "queue-active-date-slice",
      onChange: () => {
        resetQueuePage();
        populateQueue();
      },
    });
    const statusSel = document.getElementById("queue-status");
    const evidenceSel = document.getElementById("queue-evidence");
    const shiftSel = document.getElementById("queue-shift");
    const sortSel = document.getElementById("queue-sort");
    const pageSizeSel = document.getElementById("queue-page-size");
    [statusSel, evidenceSel, shiftSel, sortSel, pageSizeSel].forEach((node) => {
      if (node instanceof HTMLSelectElement) {
        node.addEventListener("change", () => {
          resetQueuePage();
          populateQueue();
        });
      }
    });
    if (queuePagePrevBtn instanceof HTMLButtonElement) {
      queuePagePrevBtn.addEventListener("click", () => {
        if (queuePage <= 1) return;
        queuePage -= 1;
        populateQueue();
      });
    }
    if (queuePageNextBtn instanceof HTMLButtonElement) {
      queuePageNextBtn.addEventListener("click", () => {
        queuePage += 1;
        populateQueue();
      });
    }
    queuePageSize = readQueuePageSize();
    populateQueue();
  }
  if (body && body.classList.contains("page-event-detail")) {
    window.addEventListener("hashchange", () => {
      populateDetail();
    });
    populateDetail();
  }
  if (body && body.classList.contains("page-setup")) {
    populateSetup();
  }
  if (body && body.classList.contains("page-dashboard")) {
    bindStaleIndicator({ pillId: "stale-pill-dashboard", onRefresh: () => populateDashboard() });
    bindDateControls({
      fromId: "date-from",
      toId: "date-to",
      rangeId: "range",
      applyId: "date-apply",
      clearId: "date-clear",
      labelId: "dashboard-active-date-slice",
      onChange: () => populateDashboard(),
    });
    populateDashboard();
  }
})();
