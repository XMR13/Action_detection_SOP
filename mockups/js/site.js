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

  const displayReviewSource = (raw) => {
    const v = String(raw || "PENDING").toUpperCase();
    if (v === "AUTO") return "AUTO";
    if (v === "MANUAL") return "MANUAL";
    return "PENDING";
  };

  const pillClassForReviewSource = (raw) => {
    const v = String(raw || "PENDING").toUpperCase();
    if (v === "AUTO") return "auto";
    if (v === "MANUAL") return "ink";
    return "pending";
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
  const selectedMachineStatus = document.getElementById("selected-machine-status");
  const selectedHumanStatus = document.getElementById("selected-human-status");
  const selectedReviewSource = document.getElementById("selected-review-source");
  const selectedFinalStatus = document.getElementById("selected-final-status");
  const selectedEvidence = document.getElementById("selected-evidence");
  const selectedSessionSla = document.getElementById("selected-session-sla");
  const selectedDetailLink = document.getElementById("selected-detail-link");
  const queueOpenSelected = document.getElementById("queue-open-selected");

  const syncReviewDock = (link) => {
    const row = link.closest("tr");
    if (!(row instanceof HTMLTableRowElement)) {
      return;
    }

    const { sessionId, sessionUid, date, shift, machine, human, reviewSource, final, evidence, sla } = row.dataset;
    const resolvedSessionId = sessionId || link.textContent?.trim() || "UNKNOWN";
    const resolvedShift = shift || "-";
    const resolvedMachine = machine || "-";
    const resolvedHuman = human || "-";
    const resolvedReviewSource = reviewSource || "-";
    const resolvedFinal = final || "-";
    const resolvedEvidence = evidence || "-";
    const resolvedSessionUid = sessionUid || "-";
    const resolvedDate = date || "-";
    const resolvedSla = sla || "-";

    if (selectedSessionIdInput instanceof HTMLInputElement) {
      selectedSessionIdInput.value = resolvedSessionId;
    }
    if (selectedSessionInline) {
      selectedSessionInline.textContent = resolvedSessionId;
    }
    if (selectedSessionLabel) {
      selectedSessionLabel.textContent = `${resolvedSessionId} | ${resolvedDate} | Machine ${resolvedMachine}`;
    }
    if (selectedSessionUid) {
      selectedSessionUid.textContent = resolvedSessionUid;
    }
    if (selectedSessionDate) {
      selectedSessionDate.textContent = resolvedDate;
    }
    if (selectedMachineStatus) {
      selectedMachineStatus.textContent = resolvedMachine;
    }
    if (selectedHumanStatus) {
      selectedHumanStatus.textContent = resolvedHuman;
    }
    if (selectedReviewSource) {
      selectedReviewSource.textContent = resolvedReviewSource;
    }
    if (selectedFinalStatus) {
      selectedFinalStatus.textContent = resolvedFinal;
    }
    if (selectedEvidence) {
      selectedEvidence.textContent = resolvedEvidence;
    }
    if (selectedSessionSla) {
      selectedSessionSla.textContent = resolvedSla === "reviewed" ? "Reviewed" : `SLA ${resolvedSla}`;
    }
    if (selectedDetailLink instanceof HTMLAnchorElement) {
      const targetUid = resolvedSessionUid && resolvedSessionUid !== "-" ? resolvedSessionUid : resolvedSessionId;
      selectedDetailLink.href = buildUiHrefWithDate("session-detail.html", encodeURIComponent(String(targetUid)));
    }
    if (queueOpenSelected instanceof HTMLAnchorElement) {
      const targetUid = resolvedSessionUid && resolvedSessionUid !== "-" ? resolvedSessionUid : resolvedSessionId;
      queueOpenSelected.href = buildUiHrefWithDate("session-detail.html", encodeURIComponent(String(targetUid)));
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
        queueLengthHint.textContent = `Queue length: ${pending}`;
      }
      const pendingPill = document.getElementById("queue-pending-pill");
      if (pendingPill && pending != null && !Number.isNaN(pending)) {
        pendingPill.textContent = `pending ${pending}`;
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
    const sortSel = document.getElementById("queue-sort");

    const reviewStatus = statusSel instanceof HTMLSelectElement ? String(statusSel.value || "") : "";
    const evidenceFilter = evidenceSel instanceof HTMLSelectElement ? String(evidenceSel.value || "") : "";
    const sort = sortSel instanceof HTMLSelectElement ? String(sortSel.value || "NEWEST") : "NEWEST";

    let url = withDateApiQuery(`/api/sessions?limit=200&sort=${encodeURIComponent(sort || "NEWEST")}`);
    if (reviewStatus && reviewStatus !== "ALL") {
      url += `&review_status=${encodeURIComponent(reviewStatus)}`;
    }

    let payload;
    try {
      payload = await apiFetchJson(url);
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="10"><span class="pill no">Failed to load sessions</span></td></tr>`;
      if (queueBody && queueBody.classList.contains("page-review-queue")) {
        queueBody.classList.remove("is-hydrating");
      }
      return;
    }

    let sessions = Array.isArray(payload.sessions) ? payload.sessions : [];
    if (evidenceFilter === "CLIP_THUMB") {
      sessions = sessions.filter((s) => Number(s.clip_count || 0) > 0 && Boolean(s.has_thumbnail));
    } else if (evidenceFilter === "CLIP_ONLY") {
      sessions = sessions.filter((s) => Number(s.clip_count || 0) > 0 && !Boolean(s.has_thumbnail));
    } else if (evidenceFilter === "THUMB_ONLY") {
      sessions = sessions.filter((s) => Number(s.clip_count || 0) <= 0 && Boolean(s.has_thumbnail));
    }
    if (sessions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10"><span class="pill">No sessions found</span></td></tr>`;
      if (queueBody && queueBody.classList.contains("page-review-queue")) {
        queueBody.classList.remove("is-hydrating");
      }
      return;
    }

    const rowsHtml = sessions
      .map((s, index) => {
        const uid = String(s.session_uid || "");
        const sid = String(s.session_id || uid || "-");
        const machine = String(s.machine_sop || s.machine_helmet || "UNKNOWN");
        const review = String(s.review_status || "PENDING");
        const reviewSource = String(s.review_source || "PENDING");
        const final = String(s.final_sop || s.final_helmet || machine);
        const date = String(s.date || "-");
        const start = formatHmsFromIso(s.start_time_iso);
        const dur = formatDuration(s.duration_s);
        const thumbUrl = s.thumbnail_url
          ? String(s.thumbnail_url)
          : s.has_thumbnail
          ? `/media/${encodeURIComponent(uid)}/thumbnail.jpg`
          : "";
        const evidenceLabel =
          s.clip_count > 0 ? (s.has_thumbnail ? "thumb + clip" : "clip") : s.has_thumbnail ? "thumbnail" : "-";
        const evidenceBadge = s.clip_count > 0 ? `<span class="queue-evidence-badge" title="${s.clip_count} clip(s)">▶</span>` : "";
        const sla = review === "PENDING" ? "-" : "reviewed";

        const rowActive = index === 0 ? " queue-row-active" : "";
        const linkActive = index === 0 ? " active" : "";

        return `
          <tr class="${rowActive.trim()}"
            data-session-id="${sid}"
            data-session-uid="${uid}"
            data-date="${date}"
            data-shift="-"
            data-machine="${displayStepStatus(machine)}"
            data-human="${displayReviewStatus(review)}"
            data-review-source="${displayReviewSource(reviewSource)}"
            data-final="${displayStepStatus(final)}"
            data-evidence="${evidenceLabel}"
            data-sla="${sla}"
          >
            <td>
              <a class="queue-session-link${linkActive}" href="#${encodeURIComponent(uid)}" data-queue-link>${sid}</a>
              <div class="table-sub">${uid}</div>
            </td>
            <td>${date}</td>
            <td>${start}</td>
            <td>${dur}</td>
            <td><span class="pill ${pillClassForStepStatus(machine)}">${displayStepStatus(machine)}</span></td>
            <td><div class="queue-review-cell"><span class="pill ${pillClassForReviewStatus(review)}">${displayReviewStatus(review)}</span><span class="pill ${pillClassForReviewSource(reviewSource)}">${displayReviewSource(reviewSource)}</span></div></td>
            <td><span class="pill ${pillClassForStepStatus(final)}">${displayStepStatus(final)}</span></td>
            <td>
              <div class="queue-evidence-cell">
                <div class="queue-evidence-preview ${thumbUrl ? "" : "queue-evidence-empty"}">
                  ${thumbUrl ? `<img class="queue-evidence-thumb" alt="session evidence thumbnail" src="${thumbUrl}" loading="lazy" />` : `<span class="queue-evidence-none">No preview</span>`}
                  ${evidenceBadge}
                </div>
              </div>
            </td>
            <td><span class="pill">${sla}</span></td>
            <td><a class="btn btn-compact action-inspect" href="${buildUiHrefWithDate("session-detail.html", encodeURIComponent(uid))}">Inspect</a></td>
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
    const detailDateLabel = document.getElementById("detail-active-date-slice");
    if (detailDateLabel) detailDateLabel.textContent = dateSliceLabel();
    const hash = window.location.hash ? window.location.hash.slice(1) : "";
    const sessionUid = hash ? decodeURIComponent(hash) : "";
    if (!sessionUid) {
      // If user opens Session Detail from the sidebar, pick the newest session.
      try {
        const list = await apiFetchJson(withDateApiQuery("/api/sessions?limit=1"));
        const sessions = Array.isArray(list.sessions) ? list.sessions : [];
        if (sessions.length > 0 && sessions[0].session_uid) {
          window.location.hash = encodeURIComponent(String(sessions[0].session_uid));
          return;
        }
      } catch (err) {
        // ignore
      }
      const playerEmpty = document.querySelector(".detail-player");
      if (playerEmpty instanceof HTMLElement) {
        playerEmpty.innerHTML = '<div class="frame-overlay"><span class="pill">no session found</span></div>';
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
    const uidHint = document.getElementById("detail-session-uid-hint");
    const dateHint = document.getElementById("detail-session-date-hint");
    const sidNode = document.getElementById("detail-session-id");
    if (sidNode) sidNode.textContent = sessionId;
    if (uidHint) uidHint.textContent = `UID: ${sessionUid}`;
    if (dateHint) dateHint.textContent = `Session date: ${sessionDate}`;
    if (selectedSessionIdInput instanceof HTMLInputElement) {
      selectedSessionIdInput.value = sessionId;
    }

    const metaSessionId = document.getElementById("detail-meta-session-id");
    const metaSessionUid = document.getElementById("detail-meta-session-uid");
    const metaSessionDate = document.getElementById("detail-meta-session-date");
    const metaStartTime = document.getElementById("detail-meta-start-time");
    const metaDuration = document.getElementById("detail-meta-duration");
    if (metaSessionId) metaSessionId.textContent = sessionId;
    if (metaSessionUid) metaSessionUid.textContent = sessionUid;
    if (metaSessionDate) metaSessionDate.textContent = sessionDate;
    if (metaStartTime) {
      const startIso = payload.checklist && payload.checklist.start_time_iso ? String(payload.checklist.start_time_iso) : "";
      metaStartTime.textContent = formatDateTimeFromIso(startIso);
    }
    if (metaDuration) {
      const startS =
        payload.checklist && payload.checklist.start_time_s != null ? Number(payload.checklist.start_time_s) : Number.NaN;
      const endS =
        payload.checklist && payload.checklist.end_time_s != null ? Number(payload.checklist.end_time_s) : Number.NaN;
      if (!Number.isNaN(startS) && !Number.isNaN(endS) && endS >= startS) {
        metaDuration.textContent = `${(endS - startS).toFixed(1)} seconds`;
      } else {
        metaDuration.textContent = "-";
      }
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
        nextLink.href = buildUiHrefWithDate("session-detail.html", encodeURIComponent(nextUid));
        nextLink.textContent = "Next Pending";
      } else {
        nextLink.href = buildUiHrefWithDate("review-queue.html");
        nextLink.textContent = "No Pending";
      }
    }

    const machine = String(payload.machine_sop || payload.machine_helmet || "UNKNOWN");
    const review = String(payload.review_status || (payload.review && payload.review.review_status) || "PENDING");
    const reviewSource = String(payload.review_source || "PENDING");
    const autoReason = payload.auto_review_reason ? String(payload.auto_review_reason) : "";
    const machinePill = document.getElementById("detail-machine-status");
    const reviewPill = document.getElementById("detail-review-status");
    const reviewSourcePill = document.getElementById("detail-review-source");
    const autoReasonNode = document.getElementById("detail-auto-review-reason");
    if (machinePill) {
      machinePill.className = `pill ${pillClassForStepStatus(machine)}`;
      machinePill.textContent = `machine ${displayStepStatus(machine)}`;
    }
    if (reviewPill) {
      reviewPill.className = `pill ${pillClassForReviewStatus(review)}`;
      reviewPill.textContent = `review ${displayReviewStatus(review)}`;
    }
    if (reviewSourcePill) {
      reviewSourcePill.className = `pill ${pillClassForReviewSource(reviewSource)}`;
      reviewSourcePill.textContent = `source ${displayReviewSource(reviewSource)}`;
    }
    if (autoReasonNode) {
      if (reviewSource.toUpperCase() === "AUTO" && autoReason) {
        autoReasonNode.textContent = `Auto decision policy: ${autoReason}`;
      } else if (reviewSource.toUpperCase() === "PENDING" && autoReason) {
        autoReasonNode.textContent = `Pending policy reason: ${autoReason}`;
      } else if (reviewSource.toUpperCase() === "MANUAL") {
        autoReasonNode.textContent = "Manual review submitted.";
      } else {
        autoReasonNode.textContent = "Auto decision policy: -";
      }
    }

    const noteBox = document.getElementById("review-note");
    if (noteBox instanceof HTMLTextAreaElement) {
      noteBox.value = payload.review && payload.review.review_note ? String(payload.review.review_note) : "";
    }
    const overrideHelmet = document.getElementById("override-helmet");
    if (overrideHelmet instanceof HTMLSelectElement) {
      const ov = payload.review && payload.review.overrides ? payload.review.overrides : {};
      const helmetOv = ov && ov.helmet ? String(ov.helmet) : "";
      overrideHelmet.value = helmetOv;
    }

    const player = document.querySelector(".detail-player");
    const evidenceStrip = document.getElementById("detail-evidence-strip");
    const evidenceState = document.getElementById("detail-evidence-state");
    if (player instanceof HTMLElement) {
      const renderThumbnail = () => {
        if (payload.thumbnail_url) {
          const url = String(payload.thumbnail_url);
          player.innerHTML = `<img alt="thumbnail" src="${url}" style="width:100%;height:100%;object-fit:cover;border-radius:inherit;" />`;
          return;
        }
        player.innerHTML = '<div class="frame-overlay"><span class="pill">no clip</span></div>';
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
      const finalSopStatus = String(payload.final_sop || payload.machine_sop || payload.final_helmet || payload.machine_helmet || "UNKNOWN").toUpperCase();
      const preferAnnotated = hasAnnotated && finalSopStatus === "UNKNOWN";

      const setEvidenceState = () => {
        if (!(evidenceState instanceof HTMLElement)) return;
        if (preferAnnotated) {
          evidenceState.className = "pill dir-b";
          evidenceState.textContent = "unknown: full annotated session";
          return;
        }
        if (clips.length > 0) {
          evidenceState.className = "pill brand";
          evidenceState.textContent = `${clips.length} clip${clips.length === 1 ? "" : "s"} available`;
          return;
        }
        if (hasAnnotated) {
          evidenceState.className = "pill brand";
          evidenceState.textContent = "annotated full session";
          return;
        }
        if (payload.thumbnail_url) {
          evidenceState.className = "pill dir-b";
          evidenceState.textContent = "thumbnail only";
          return;
        }
        evidenceState.className = "pill no";
        evidenceState.textContent = "no evidence";
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
          '<video id="detail-video" controls preload="metadata" playsinline style="width:100%;height:100%;object-fit:cover;border-radius:inherit;"></video><div class="frame-overlay"><a id="detail-clip-link" class="pill" target="_blank" rel="noreferrer">open clip file</a></div>';
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
          noClipPill.textContent = "no clip events";
          evidenceStrip.appendChild(noClipPill);
        }

        if (hasAnnotated) {
          const annotatedBtn = document.createElement("button");
          annotatedBtn.type = "button";
          annotatedBtn.className = "pill detail-clip-pill";
          annotatedBtn.setAttribute("data-clip-key", "annotated");
          annotatedBtn.textContent = "annotated full session";
          annotatedBtn.addEventListener("click", () => {
            renderVideoClip(annotatedPlaybackUrl || annotatedUrl, annotatedUrl);
            setActiveClipPill("annotated");
          });
          evidenceStrip.appendChild(annotatedBtn);
        }

        const artifactsLink = document.createElement("a");
        artifactsLink.className = "pill";
        artifactsLink.href = "#artifacts";
        artifactsLink.textContent = "open artifacts";
        evidenceStrip.appendChild(artifactsLink);
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

    const checklistWrap = document.querySelector(".checklist");
    if (checklistWrap instanceof HTMLElement && payload.checklist) {
      const checklist = payload.checklist;
      const overrides = payload.review && payload.review.overrides ? payload.review.overrides : {};
      const steps = [
        { key: "operator_present", label: "Operator present in ROI" },
        { key: "roi_dwell", label: "ROI dwell" },
        { key: "helmet", label: "Helmet compliance" },
      ];
      checklistWrap.innerHTML = steps
        .map((step) => {
          const machineVal = String(checklist[step.key] || "UNKNOWN");
          const reviewVal = overrides && overrides[step.key] ? String(overrides[step.key]) : machineVal;
          return `
            <div class="step-card">
              <div class="step-head">
                <strong>${step.label}</strong>
                <div class="step-pills">
                  <span class="pill ${pillClassForStepStatus(machineVal)}">machine ${displayStepStatus(machineVal)}</span>
                  <span class="pill ${pillClassForStepStatus(reviewVal)}">review ${displayStepStatus(reviewVal)}</span>
                </div>
              </div>
            </div>
          `;
        })
        .join("");
    }

    const artifactsList = document.querySelector(".artifacts-list");
    if (artifactsList instanceof HTMLElement) {
      const artifacts = Array.isArray(payload.artifacts) ? payload.artifacts : [];
      artifactsList.innerHTML = artifacts
        .map((a) => {
          const name = a && a.name ? String(a.name) : "-";
          const url = a && a.url ? String(a.url) : "";
          const safeUrl = url ? url : "#";
          return `
            <a class="artifact-item" href="${safeUrl}" target="_blank" rel="noreferrer">
              <strong>${name}</strong>
              <span>Open artifact</span>
            </a>
          `;
        })
        .join("");
    }

    if (form) {
      form.onsubmit = async (event) => {
        event.preventDefault();
        const reviewStatus = statusInput instanceof HTMLInputElement ? statusInput.value : "PENDING";
        const note = noteBox instanceof HTMLTextAreaElement ? noteBox.value : "";
        const overrides = {};
        if (overrideHelmet instanceof HTMLSelectElement && overrideHelmet.value) {
          overrides.helmet = overrideHelmet.value;
        }
        try {
          await apiFetchJson(`/api/sessions/${encodeURIComponent(sessionUid)}/review`, {
            method: "PUT",
            body: JSON.stringify({ review_status: reviewStatus, review_note: note, overrides }),
          });
          if (reviewPill) {
            reviewPill.className = `pill ${pillClassForReviewStatus(reviewStatus)}`;
            reviewPill.textContent = `review ${displayReviewStatus(reviewStatus)}`;
          }
          if (reviewSourcePill) {
            reviewSourcePill.className = `pill ${pillClassForReviewSource("MANUAL")}`;
            reviewSourcePill.textContent = "source MANUAL";
          }
          if (autoReasonNode) {
            autoReasonNode.textContent = "Manual review submitted.";
          }
          if (String(reviewStatus || "").toUpperCase() !== "PENDING") {
            const nextUid = await resolveNextPending();
            if (nextUid && nextUid !== sessionUid) {
              window.location.assign(buildUiHrefWithDate("session-detail.html", encodeURIComponent(nextUid)));
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
    const lastScan = document.getElementById("admin-last-scan");
    const sessionCount = document.getElementById("admin-session-count");
    const rescanBtn = document.getElementById("admin-rescan");

    const refresh = async () => {
      try {
        const cfg = await apiFetchJson("/api/config");
        if (dataDir) dataDir.textContent = String(cfg.data_dir || "-");
        if (lastScan) lastScan.textContent = String(cfg.last_scan_utc || "-");
        if (sessionCount) sessionCount.textContent = String(cfg.session_count ?? "-");
      } catch (err) {
        if (dataDir) dataDir.textContent = "Failed to load";
      }
    };

    await refresh();

    if (rescanBtn instanceof HTMLButtonElement) {
      rescanBtn.addEventListener("click", async () => {
        rescanBtn.setAttribute("disabled", "disabled");
        try {
          const res = await apiFetchJson("/api/admin/rescan", { method: "POST" });
          if (lastScan) lastScan.textContent = String(res.last_scan_utc || "-");
          if (sessionCount) sessionCount.textContent = String(res.session_count ?? "-");
        } catch (err) {
          alert("Rescan failed");
        } finally {
          rescanBtn.removeAttribute("disabled");
        }
      });
    }
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
        totalHint.textContent = `approved ${String(s.approved ?? 0)} | rejected ${String(s.rejected ?? 0)}`;
      }

      const pendingHint = document.getElementById("kpi-pending-hint");
      if (pendingHint) {
        const reviewed = Number(s.reviewed ?? 0);
        const total = Number(s.total_sessions ?? 0);
        const completionPct = Number(s.review_completion_pct ?? 0);
        pendingHint.textContent =
          total > 0
            ? `${reviewed}/${total} decided (${completionPct.toFixed(1)}%)`
            : "No sessions yet";
      }

      const bannerPending = document.getElementById("banner-pending-pill");
      if (bannerPending) bannerPending.textContent = `pending ${String(s.pending ?? "-")}`;

      const helmetCheckedNode = document.getElementById("kpi-helmet-checked");
      if (helmetCheckedNode) helmetCheckedNode.textContent = String(s.final_helmet_done ?? "-");
      const helmetCheckedHint = document.getElementById("kpi-helmet-checked-hint");
      if (helmetCheckedHint) {
        const pct = Number(s.reviewed_final_done_pct ?? 0);
        helmetCheckedHint.textContent = `${pct.toFixed(1)}% DONE across reviewed sessions`;
      }

      const unknownRateNode = document.getElementById("kpi-unknown-rate");
      if (unknownRateNode) {
        const pct = Number(s.final_unknown_pct ?? 0);
        unknownRateNode.textContent = `${pct.toFixed(1)}%`;
      }
      const unknownRateHint = document.getElementById("kpi-unknown-rate-hint");
      if (unknownRateHint) {
        unknownRateHint.textContent = `${String(s.final_helmet_unknown ?? 0)} UNKNOWN from final helmet status`;
      }

      const manualReviewNode = document.getElementById("kpi-manual-review");
      if (manualReviewNode) manualReviewNode.textContent = String(s.human_reviewed ?? s.reviewed ?? "-");
      const manualReviewHint = document.getElementById("kpi-manual-review-hint");
      if (manualReviewHint) {
        manualReviewHint.textContent = `${String(s.manual_helmet_overrides ?? 0)} helmet overrides`;
      }

      const compactManualNeeded = document.getElementById("dashboard-compact-manual-needed");
      if (compactManualNeeded) compactManualNeeded.textContent = String(s.final_helmet_unknown ?? "-");
      const compactApproved = document.getElementById("dashboard-compact-approved");
      if (compactApproved) compactApproved.textContent = String(s.approved ?? "-");
      const compactMachineNo = document.getElementById("dashboard-compact-machine-no");
      if (compactMachineNo) compactMachineNo.textContent = String(s.machine_helmet_not_done ?? "-");
      const compactPending = document.getElementById("dashboard-compact-pending");
      if (compactPending) compactPending.textContent = String(s.pending ?? "-");

      const trendDone = document.getElementById("trend-strip-done");
      if (trendDone) {
        trendDone.textContent = `done ${String(s.final_helmet_done ?? 0)}`;
      }
      const trendUnknown = document.getElementById("trend-strip-unknown");
      if (trendUnknown) {
        const unknownPct = Number(s.final_unknown_pct ?? 0);
        trendUnknown.textContent = `unknown ${String(s.final_helmet_unknown ?? 0)} (${unknownPct.toFixed(1)}%)`;
      }
      const trendReview = document.getElementById("trend-strip-review");
      if (trendReview) {
        const completionPct = Number(s.review_completion_pct ?? 0);
        trendReview.textContent = `reviewed ${String(s.reviewed ?? 0)}/${String(s.total_sessions ?? 0)} (${completionPct.toFixed(1)}%)`;
      }

      const renderDashboardTrend = async () => {
        const svg = document.querySelector(".trend-svg");
        if (!(svg instanceof SVGSVGElement)) return;

        const donePath = svg.querySelector("path.trend-line.dir-a");
        const unknownPath = svg.querySelector("path.trend-line.dir-b");
        const doneDot = svg.querySelector("circle.trend-focus-dot.dir-a");
        const unknownDot = svg.querySelector("circle.trend-focus-dot.dir-b");
        const gridValues = Array.from(svg.querySelectorAll("text.trend-grid-value"));

        if (!(donePath instanceof SVGPathElement) || !(unknownPath instanceof SVGPathElement)) return;

        const slice = readDateSliceFromUrl();

        let listPayload;
        try {
          let url = "";
          if (slice.from || slice.to) {
            url = withDateApiQuery("/api/sessions?limit=2000&sort=OLDEST");
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
            url = activeDate
              ? `/api/sessions?limit=2000&sort=OLDEST&date=${encodeURIComponent(activeDate)}`
              : "/api/sessions?limit=2000&sort=OLDEST";
          }
          listPayload = await apiFetchJson(url);
        } catch (err) {
          return;
        }

        const sessions = Array.isArray(listPayload.sessions) ? listPayload.sessions : [];
        const shiftStartHour = 6;
        const shiftEndHour = 18;
        const binCount = shiftEndHour - shiftStartHour + 1; // inclusive hours
        const done = Array(binCount).fill(0);
        const unknown = Array(binCount).fill(0);

        sessions.forEach((row) => {
          const iso = row && row.start_time_iso ? String(row.start_time_iso) : row && row.end_time_iso ? String(row.end_time_iso) : "";
          if (!iso) return;
          const dt = new Date(iso);
          if (Number.isNaN(dt.getTime())) return;
          const h = dt.getHours();
          if (h < shiftStartHour || h > shiftEndHour) return;
          const idx = h - shiftStartHour;

          const status = String(row.final_helmet || row.machine_helmet || "UNKNOWN").toUpperCase();
          if (status === "DONE") done[idx] += 1;
          else if (status === "UNKNOWN") unknown[idx] += 1;
        });

        const maxVal = Math.max(1, ...done, ...unknown);
        const view = svg.viewBox && svg.viewBox.baseVal ? svg.viewBox.baseVal : { x: 0, y: 0, width: 760, height: 190 };
        const xMin = 60;
        const xMax = 720;
        const yTop = 26;
        const yBottom = 152;
        const xStep = binCount > 1 ? (xMax - xMin) / (binCount - 1) : 0;
        const yScale = (v) => yBottom - (Math.max(0, Number(v || 0)) / maxVal) * (yBottom - yTop);

        const ptsDone = done.map((v, i) => ({ x: xMin + i * xStep, y: yScale(v) }));
        const ptsUnknown = unknown.map((v, i) => ({ x: xMin + i * xStep, y: yScale(v) }));

        const doneD = buildSmoothPath(ptsDone);
        const unknownD = buildSmoothPath(ptsUnknown);
        donePath.setAttribute("d", doneD || `M${xMin} ${yBottom} L${xMax} ${yBottom}`);
        unknownPath.setAttribute("d", unknownD || `M${xMin} ${yBottom} L${xMax} ${yBottom}`);

        const lastDone = ptsDone[ptsDone.length - 1];
        const lastUnknown = ptsUnknown[ptsUnknown.length - 1];
        if (doneDot instanceof SVGCircleElement) {
          doneDot.setAttribute("cx", String(lastDone.x.toFixed(1)));
          doneDot.setAttribute("cy", String(lastDone.y.toFixed(1)));
        }
        if (unknownDot instanceof SVGCircleElement) {
          unknownDot.setAttribute("cx", String(lastUnknown.x.toFixed(1)));
          unknownDot.setAttribute("cy", String(lastUnknown.y.toFixed(1)));
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
        const doneAt = new Date();
        doneAt.setHours(shiftStartHour + donePeak.idx, 0, 0, 0);
        const unknownAt = new Date();
        unknownAt.setHours(shiftStartHour + unknownPeak.idx, 0, 0, 0);

        const trendDone = document.getElementById("trend-strip-done");
        if (trendDone) trendDone.textContent = `done peak ${Math.max(0, donePeak.val)} @ ${formatHmLocal(doneAt)}`;
        const trendUnknown = document.getElementById("trend-strip-unknown");
        if (trendUnknown) trendUnknown.textContent = `unknown spike ${Math.max(0, unknownPeak.val)} @ ${formatHmLocal(unknownAt)}`;
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
                const roi = String(session.machine_roi_dwell || "UNKNOWN");
                const helmet = String(session.final_helmet || session.machine_helmet || "UNKNOWN");
                const review = String(session.review_status || "PENDING");
                const remark =
                  Number(session.clip_count || 0) > 0
                    ? `${String(session.clip_count)} clip(s) attached`
                    : session.has_thumbnail
                    ? "Thumbnail available"
                    : "No evidence attached";
                return `
                  <tr>
                    <td>
                      <div class="session-cell">
                        <span class="thumb" aria-hidden="true"></span>
                        <div>
                          <strong>${sid}</strong>
                          <div class="table-sub">${uid}</div>
                        </div>
                      </div>
                    </td>
                    <td>${start}</td>
                    <td><span class="pill ${pillClassForStepStatus(roi)}">ROI ${displayStepStatus(roi)}</span> <span class="pill ${pillClassForStepStatus(helmet)}">helmet ${displayStepStatus(helmet)}</span></td>
                    <td><span class="pill ${pillClassForReviewStatus(review)}">${displayReviewStatus(review)}</span></td>
                    <td>${remark}</td>
                    <td><a class="btn btn-compact action-inspect" href="${buildUiHrefWithDate("session-detail.html", encodeURIComponent(uid))}">Inspect</a></td>
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
    bindDateControls({
      fromId: "queue-date-from",
      toId: "queue-date-to",
      applyId: "queue-date-apply",
      clearId: "queue-date-clear",
      labelId: "queue-active-date-slice",
      onChange: () => populateQueue(),
    });
    const statusSel = document.getElementById("queue-status");
    const evidenceSel = document.getElementById("queue-evidence");
    const sortSel = document.getElementById("queue-sort");
    const shiftSel = document.getElementById("queue-shift");
    [statusSel, evidenceSel, sortSel, shiftSel].forEach((node) => {
      if (node instanceof HTMLSelectElement) {
        node.addEventListener("change", () => {
          populateQueue();
        });
      }
    });
    populateQueue();
  }
  if (body && body.classList.contains("page-event-detail")) {
    bindDateControls({
      fromId: "detail-date-from",
      toId: "detail-date-to",
      applyId: "detail-date-apply",
      clearId: "detail-date-clear",
      labelId: "detail-active-date-slice",
      onChange: () => populateDetail(),
    });
    window.addEventListener("hashchange", () => {
      populateDetail();
    });
    populateDetail();
  }
  if (body && body.classList.contains("page-setup")) {
    populateSetup();
  }
  if (body && body.classList.contains("page-dashboard")) {
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
