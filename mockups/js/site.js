import {
  formatDateTimeFromIso,
  formatHmsFromIso,
  formatDuration,
  normalizeShiftId,
  shiftLabel,
  shiftDayHourIndex,
  escapeHtml,
 } from "./shared/format.js";

import {
  toYmdLocal,
  readDateSliceFromUrl,
  dateSliceLabel,
  withDateApiQuery,
  buildUiHrefWithDate,
  buildSessionDetailHref,
  buildAlertDetailHref,
  readSessionUidFromUrl,
  readAlertUidFromUrl,
  applyDateSliceToStaticNav,
  bindDateControls,
} from "./shared/dates.js";

import { apiFetchJson } from "./shared/api.js";

import {
  readReviewContext,
  writeReviewContext,
  withReviewFilters,
  createPendingReviewNavigation,
} from "./shared/review-navigation.js";

import { initAuthUi } from "./shared/auth.js";

import { initSetupPage } from "./pages/setup.js";
import { mountSessionEvidence } from "./shared/evidence-player.js";

import {
  displayStepStatus,
  pillClassForStepStatus,
  operatorVerdict,
  verdictLabel,
  verdictClass,
  reviewSourceLabel,
  pillClassForReviewStatus,
  displayAlertStatus,
  pillClassForAlertStatus,
  structuredSop,
  sopScope,
  sopStatusValue,
  dashboardTrendStatus,
  rollOverallDisplay,
  queueDecision,
} from "./shared/status.js";

(function () {
  const countFormat = new Intl.NumberFormat("id-ID", { maximumFractionDigits: 0 });
  const percentFormat = new Intl.NumberFormat("id-ID", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const renderDashboardSummary = (stats, state = "ready") => {
    const ready = state === "ready" && stats != null;
    const count = (key) => {
      const value = ready ? stats[key] : null;
      return value != null && Number.isFinite(Number(value)) ? countFormat.format(Number(value)) : "—";
    };
    const text = (id, value) => {
      const node = document.getElementById(id);
      if (node) node.textContent = value;
    };
    document.querySelectorAll("[data-session-stat]").forEach((node) => {
      node.textContent = count(node.dataset.sessionStat);
    });
    document.querySelectorAll("[data-session-summary]").forEach((node) => {
      node.setAttribute("aria-busy", state === "loading" ? "true" : "false");
      node.dataset.state = state;
    });
    document.querySelectorAll(".session-summary-feedback").forEach((node) => {
      node.textContent = state === "error" ? "Ringkasan gagal dimuat. Pilih Coba lagi untuk memuat ulang." :
        state === "loading" ? "Memuat ringkasan…" : "";
    });
    const retry = document.getElementById("dashboard-summary-retry");
    if (retry) retry.hidden = state !== "error";
    const slice = readDateSliceFromUrl();
    document.querySelectorAll("a[data-summary-verdict]").forEach((node) => {
      // These totals cover every shift/evidence type in the selected date range.
      // Do not carry narrower table filters or its page into summary links.
      const params = new URLSearchParams();
      if (slice.from) params.set("date_from", slice.from);
      if (slice.to) params.set("date_to", slice.to);
      params.set("verdict", node.dataset.summaryVerdict);
      node.href = `review-queue.html?${params}`;
    });
    const pending = count("verdict_needs_review");
    text("banner-pending-pill", `perlu ditinjau ${pending}`);
    const equation = ready ?
      `${pending} perlu ditinjau + ${count("verdict_done")} sesuai + ${count("verdict_not_done")} tidak sesuai + ${count("out_of_scope")} di luar cakupan = ${count("total_sessions")} sesi` :
      state === "loading" ? "Memuat ringkasan…" : "Ringkasan belum tersedia";
    text("kpi-total-hint", equation);
    let completionHint = state === "loading" ? "Memuat statistik…" : "Ringkasan belum tersedia";
    if (ready) {
      const total = Number(stats.total_sessions ?? 0);
      const resolved = Number(stats.verdict_done ?? 0) + Number(stats.verdict_not_done ?? 0) + Number(stats.out_of_scope ?? 0);
      const completionPct = Number(stats.review_completion_pct ?? (total > 0 ? 100 * resolved / total : 0));
      completionHint = total > 0
        ? `${countFormat.format(resolved)}/${countFormat.format(total)} sesi sudah memiliki hasil akhir (${percentFormat.format(completionPct)}%)`
        : "Belum ada sesi";
    }
    text("kpi-pending-hint", completionHint);
    const compliance = ready ? stats.compliance_pct : null;
    text("kpi-compliance", `Kepatuhan: ${compliance != null && Number.isFinite(Number(compliance)) ? percentFormat.format(Number(compliance)) + "% dari sesi yang dinilai" : "—"}`);
  };

  const dashboardSopPills = (session) => {
    const verdict = operatorVerdict(session);
    return `<span class="pill ${verdictClass(verdict)}">${verdictLabel(verdict)}</span>`;
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
          label: "Pembersihan",
          machineValue: machine.cleaned || "UNKNOWN",
          finalValue: final.cleaned || "UNKNOWN",
          overrideValue: reviewOverrides.cleaned || "",
          values: ["DONE", "NOT_DONE", "UNKNOWN"],
          displayValue: displayStepStatus,
        }),
        renderOverrideSelect({
          key: "labeled",
          label: "Pemberian label",
          machineValue: machine.labeled || "UNKNOWN",
          finalValue: final.labeled || "UNKNOWN",
          overrideValue: reviewOverrides.labeled || "",
          values: ["DONE", "NOT_DONE", "UNKNOWN"],
          displayValue: displayStepStatus,
        }),
        renderOverrideSelect({
          key: "overall_status",
          label: "Hasil SOP akhir",
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
            <p class="caption">Ubah hasil langkah atau hasil akhir jika bukti berbeda dari penilaian AI.</p>
          </div>
          ${inconsistent}
        </div>
        <div class="sop-override-grid">${rows}</div>
      `;
      return;
    }

    const machine = sopScope(payload, "machine");
    const final = sopScope(payload, "final");
    const archiveRows = [
      ["operator_present", "Operator present"],
      ["roi_dwell", "ROI dwell"],
      ["helmet", "Helmet"],
    ].map(([key, label]) => `
      <div class="sop-override-row">
        <div>
          <span class="sop-override-label">${escapeHtml(label)}</span>
          <span class="sop-override-machine">AI: ${escapeHtml(displayStepStatus(machine[key] || "UNKNOWN"))}</span>
          <span class="sop-override-final">Final: ${escapeHtml(displayStepStatus(final[key] || "UNKNOWN"))}</span>
        </div>
      </div>
    `).join("");
    panel.innerHTML = `
      <div class="detail-sop-head">
        <div>
          <strong>Arsip sesi operator</strong>
          <p class="caption">Sesi lama hanya dapat dibaca. Bukti dan hasil tinjauan sebelumnya tetap tersedia.</p>
        </div>
      </div>
      <div class="sop-override-grid">${archiveRows}</div>
    `;
  };

  const formatHmLocal = (d) => {
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    return `${hh}:${mm}`;
  };

  const buildCountPath = (points) => points.map((point, index) =>
    `${index === 0 ? "M" : "L"}${point.x.toFixed(1)} ${point.y.toFixed(1)}`
  ).join(" ");

  const shiftDateFormat = new Intl.DateTimeFormat("id-ID", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  const shortShiftDateFormat = new Intl.DateTimeFormat("id-ID", { day: "numeric", month: "short", timeZone: "UTC" });
  const displayShiftDate = (raw, short = false) => {
    const date = new Date(`${raw}T00:00:00Z`);
    return Number.isNaN(date.getTime()) ? String(raw) : (short ? shortShiftDateFormat : shiftDateFormat).format(date);
  };
  const dashboardDateRange = () => {
    const { from, to } = readDateSliceFromUrl();
    if (from && to) return from === to ? `${displayShiftDate(from)} · WIB` : `${displayShiftDate(from)} – ${displayShiftDate(to)} · WIB`;
    if (from) return `Mulai ${displayShiftDate(from)} · WIB`;
    if (to) return `Sampai ${displayShiftDate(to)} · WIB`;
    return "Semua tanggal · WIB";
  };
  const setDashboardChartState = (state, message) => {
    const chart = document.querySelector(".dashboard-trend");
    if (chart) {
      chart.dataset.state = state;
      chart.setAttribute("aria-busy", state === "loading" ? "true" : "false");
    }
    const status = document.getElementById("trend-data-status");
    if (status) status.textContent = message;
  };

  const displayAlertName = (alert) => {
    const rawType = String((alert && (alert.alert_type || alert.machine_status)) || "")
      .trim()
      .toUpperCase();
    const words = rawType
      .replace(/[_-]+/g, " ")
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean)
      .map((word) => `${word.slice(0, 1).toUpperCase()}${word.slice(1)}`)
      .join(" ");
    return words || "Safety Alert";
  };

  const displayAlertReference = (alertUid) => {
    const uid = String(alertUid || "");
    const digest = uid.match(/_([0-9a-f]{10,64})$/i);
    return digest ? digest[1].slice(0, 8).toUpperCase() : "—";
  };

  const displayCamera = (alert) => {
    const cameraId = String((alert && alert.camera_id) || "").trim();
    if (cameraId && !cameraId.includes("://")) return cameraId;
    const source = String((alert && alert.source) || "").trim();
    return source && !source.includes("://") ? source : "Camera";
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

  // Register auth handlers once; importing auth.js does not initialize them.
  initAuthUi();

  const formNode = document.getElementById("review-form");
  const form = formNode instanceof HTMLFormElement ? formNode : null;
  const statusInput = document.getElementById("review-status");
  const actionButtons = form ? form.querySelectorAll("button[data-review-status]") : [];
  const toolbar = document.querySelector(".detail-review-toolbar");
  if (toolbar instanceof HTMLElement) {
    const updateToolbarHeight = () => document.documentElement.style.setProperty("--review-toolbar-height", `${toolbar.offsetHeight}px`);
    updateToolbarHeight();
    new ResizeObserver(updateToolbarHeight).observe(toolbar);
  }
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
  const queueExportCsv = document.getElementById("queue-export-csv");
  let queuePage = 1;
  let queuePageSize = 20;

  const queuePaginationMeta = document.getElementById("queue-page-meta");
  const queuePaginationIndicator = document.getElementById("queue-page-indicator");
  const queuePagePrevBtn = document.getElementById("queue-page-prev");
  const queuePageNextBtn = document.getElementById("queue-page-next");

  let alertPage = 1;
  let alertPageSize = 20;
  const alertPaginationMeta = document.getElementById("alert-page-meta");
  const alertPaginationIndicator = document.getElementById("alert-page-indicator");
  const alertPagePrevBtn = document.getElementById("alert-page-prev");
  const alertPageNextBtn = document.getElementById("alert-page-next");
  const alertExportCsv = document.getElementById("alert-export-csv");
  const alertOpenSelected = document.getElementById("alert-open-selected");

  const readAlertPageSize = () => {
    const node = document.getElementById("alert-page-size");
    const raw = node instanceof HTMLSelectElement ? Number(node.value) : Number.NaN;
    if (raw === 10 || raw === 20) return raw;
    return 20;
  };

  const resetAlertPage = () => {
    alertPage = 1;
  };

  const buildAlertDataUrl = (baseUrl, { includePagination } = { includePagination: true }) => {
    const statusSel = document.getElementById("alert-status");
    const sortSel = document.getElementById("alert-sort");
    const pageSizeSel = document.getElementById("alert-page-size");
    const status = statusSel instanceof HTMLSelectElement ? String(statusSel.value || "PENDING") : "PENDING";
    const sort = sortSel instanceof HTMLSelectElement ? String(sortSel.value || "NEWEST") : "NEWEST";
    alertPageSize = pageSizeSel instanceof HTMLSelectElement ? readAlertPageSize() : alertPageSize;
    if (includePagination) writeReviewContext("alert", { page: alertPage, page_size: alertPageSize, status, sort });
    const params = new URLSearchParams();
    if (includePagination) {
      params.set("page", String(alertPage));
      params.set("page_size", String(alertPageSize));
    }
    params.set("status", status || "PENDING");
    params.set("sort", sort || "NEWEST");
    const query = params.toString();
    return withDateApiQuery(`${baseUrl}${query ? `?${query}` : ""}`);
  };

  const syncAlertExportHref = () => {
    if (alertExportCsv instanceof HTMLAnchorElement) {
      alertExportCsv.href = buildAlertDataUrl("/api/alerts/export.csv", { includePagination: false });
    }
  };

  const syncAlertPaginationUi = ({ total, page, pageSize, totalPages, hasPrev, hasNext }) => {
    const safeTotal = Math.max(0, Number(total || 0));
    const safePage = Math.max(1, Number(page || 1));
    const safePageSize = Math.max(1, Number(pageSize || 20));
    const safePages = Math.max(0, Number(totalPages || 0));
    const from = safeTotal > 0 ? (safePage - 1) * safePageSize + 1 : 0;
    const to = safeTotal > 0 ? Math.min(safeTotal, safePage * safePageSize) : 0;
    if (alertPaginationMeta) {
      alertPaginationMeta.textContent = safeTotal > 0 ? `Showing ${from}-${to} of ${safeTotal}` : "Showing 0";
    }
    if (alertPaginationIndicator) {
      alertPaginationIndicator.textContent = safePages > 0 ? `Page ${safePage}/${safePages}` : "Page 0/0";
    }
    if (alertPagePrevBtn instanceof HTMLButtonElement) {
      alertPagePrevBtn.disabled = !hasPrev;
      alertPagePrevBtn.setAttribute("aria-disabled", hasPrev ? "false" : "true");
    }
    if (alertPageNextBtn instanceof HTMLButtonElement) {
      alertPageNextBtn.disabled = !hasNext;
      alertPageNextBtn.setAttribute("aria-disabled", hasNext ? "false" : "true");
    }
  };

  const readQueuePageSize = () => {
    const node = document.getElementById("queue-page-size");
    const raw = node instanceof HTMLSelectElement ? Number(node.value) : Number.NaN;
    if (raw === 10 || raw === 20) return raw;
    return 20;
  };

  const resetQueuePage = () => {
    queuePage = 1;
  };

  const buildQueueDataUrl = (baseUrl, { includePagination } = { includePagination: true }) => {
    const statusSel = document.getElementById("queue-status");
    const evidenceSel = document.getElementById("queue-evidence");
    const shiftSel = document.getElementById("queue-shift");
    const sortSel = document.getElementById("queue-sort");
    const pageSizeSel = document.getElementById("queue-page-size");

    const selectedVerdict = statusSel instanceof HTMLSelectElement ? String(statusSel.value || "") : "";
    const evidenceFilter = evidenceSel instanceof HTMLSelectElement ? String(evidenceSel.value || "ANY") : "ANY";
    const shiftFilter = shiftSel instanceof HTMLSelectElement ? String(shiftSel.value || "ALL") : "ALL";
    const sort = sortSel instanceof HTMLSelectElement ? String(sortSel.value || "NEWEST") : "NEWEST";
    queuePageSize = pageSizeSel instanceof HTMLSelectElement ? readQueuePageSize() : queuePageSize;
    if (includePagination) writeReviewContext("session", {
      page: queuePage, page_size: queuePageSize, verdict: selectedVerdict,
      evidence: evidenceFilter, shift: shiftFilter, sort,
    });

    const params = new URLSearchParams();
    if (includePagination) {
      params.set("page", String(queuePage));
      params.set("page_size", String(queuePageSize));
    }
    params.set("sort", sort || "NEWEST");
    params.set("evidence", evidenceFilter || "ANY");
    params.set("shift", shiftFilter || "ALL");
    if (selectedVerdict && selectedVerdict !== "ALL") {
      params.set("operator_verdict", selectedVerdict);
    }

    const query = params.toString();
    return withDateApiQuery(`${baseUrl}${query ? `?${query}` : ""}`);
  };

  const syncQueueExportHref = () => {
    if (queueExportCsv instanceof HTMLAnchorElement) {
      queueExportCsv.href = buildQueueDataUrl("/api/sessions/export.csv", { includePagination: false });
    }
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
      selectedSessionLabel.textContent = `${resolvedSessionId} | ${resolvedDate} | ${resolvedShift} | Hasil akhir: ${resolvedFinal}`;
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
      selectedDetailLink.hidden = false;
      const targetUid = resolvedSessionUid && resolvedSessionUid !== "-" ? resolvedSessionUid : resolvedSessionId;
      selectedDetailLink.href = buildSessionDetailHref(String(targetUid));
    }
    if (queueOpenSelected instanceof HTMLAnchorElement) {
      queueOpenSelected.hidden = false;
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

    if (statusInput instanceof HTMLSelectElement && !statusInput.disabled) {
      statusInput.value = status;
      statusInput.dispatchEvent(new Event("change"));
      document.getElementById("review-save")?.focus();
    }
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

  let queueLoadVersion = 0;
  const clearQueueSelection = () => {
    queueLinks = [];
    for (const node of [selectedSessionUid, selectedSessionDate, selectedSessionShift,
      selectedMachineStatus, selectedHumanStatus, selectedFinalStatus, selectedEvidence]) {
      if (node) node.textContent = "—";
    }
    if (selectedSessionLabel) selectedSessionLabel.textContent = "Pilih sesi dari tabel untuk melihat konteksnya.";
    for (const link of [selectedDetailLink, queueOpenSelected]) {
      if (link) link.hidden = true;
    }
  };
  const setQueueLoadState = (state, total = null) => {
    const count = document.getElementById("queue-length-hint");
    if (count) count.textContent = state === "ready" ? `Hasil filter: ${countFormat.format(total)} sesi` :
      state === "loading" ? "Memuat sesi…" : "Data belum tersedia";
    const feedback = document.getElementById("queue-load-feedback");
    if (feedback) feedback.textContent = state === "error" ? "Sesi gagal dimuat. Pilih Coba lagi untuk memuat ulang." : "";
    const retry = document.getElementById("queue-load-retry");
    if (retry) retry.hidden = state !== "error";
    const panel = document.querySelector(".queue-table-panel");
    if (panel) panel.setAttribute("aria-busy", state === "loading" ? "true" : "false");
  };
  const populateQueue = async () => {
    const loadVersion = ++queueLoadVersion;
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
    setQueueLoadState("loading");
    clearQueueSelection();
    tbody.innerHTML = `<tr><td colspan="9"><span class="pill pending">Memuat sesi…</span></td></tr>`;

    syncQueueExportHref();
    const url = buildQueueDataUrl("/api/sessions", { includePagination: true });

    let payload;
    try {
      payload = await apiFetchJson(url);
      if (loadVersion !== queueLoadVersion) return;
    } catch (err) {
      if (loadVersion !== queueLoadVersion) return;
      setQueueLoadState("error");
      syncQueuePaginationUi({ total: 0, page: 1, pageSize: queuePageSize, totalPages: 0, hasPrev: false, hasNext: false });
      tbody.innerHTML = `<tr><td colspan="9"><span class="pill no">Sesi gagal dimuat</span></td></tr>`;
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
    setQueueLoadState("ready", Number(payload.total ?? sessions.length));
    syncQueuePaginationUi({
      total: Number(payload.total || sessions.length),
      page: Number(payload.page || queuePage),
      pageSize: Number(payload.page_size || queuePageSize),
      totalPages: serverTotalPages,
      hasPrev: Boolean(payload.has_prev),
      hasNext: Boolean(payload.has_next),
    });

    if (sessions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9"><span class="pill">Tidak ada sesi yang cocok dengan filter</span></td></tr>`;
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
        const decision = queueDecision(s);
        const date = String(s.date || "-");
        const shift = shiftLabel(s.shift_id, s.shift_name);
        const start = formatHmsFromIso(s.start_time_iso);
        const end = formatHmsFromIso(s.end_time_iso);
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
            data-machine="${escapeHtml(machine === "UNKNOWN" ? "Bukti belum cukup" : verdictLabel(machine))}"
            data-human="${escapeHtml(reviewSourceLabel(s.review_source))}"
            data-final="${escapeHtml(decision.label)}"
            data-evidence="${escapeHtml(evidenceLabel)}"
          >
            <td>
              <a class="queue-session-link${linkActive}" href="#${encodeURIComponent(uid)}" title="${uid ? `UID ${escapeHtml(uid)}` : ""}" data-queue-link>${escapeHtml(sid)}</a>
            </td>
            <td>${escapeHtml(date)}</td>
            <td><span class="pill ink">${escapeHtml(shift)}</span></td>
            <td>${escapeHtml(start)}</td>
            <td>${escapeHtml(end)}</td>
            <td>${escapeHtml(dur)}</td>
            <td>
              <div class="queue-decision-cell">
                <span class="pill ${decision.className}">${escapeHtml(decision.label)}</span>
                ${decision.steps ? `<div class="table-sub queue-decision-meta">${escapeHtml(decision.steps)}</div>` : ""}
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

  const populateAlerts = async () => {
    const tbody = document.querySelector(".page-helmet-alerts .queue-table tbody");
    const alertBody = document.body;
    if (!(tbody instanceof HTMLTableSectionElement)) {
      return;
    }
    if (alertBody && alertBody.classList.contains("page-helmet-alerts")) {
      alertBody.classList.add("is-hydrating");
    }
    const dateLabel = document.getElementById("alert-active-date-slice");
    if (dateLabel) dateLabel.textContent = dateSliceLabel();
    syncAlertExportHref();

    let payload;
    try {
      payload = await apiFetchJson(buildAlertDataUrl("/api/alerts"));
    } catch (err) {
      syncAlertPaginationUi({ total: 0, page: 1, pageSize: alertPageSize, totalPages: 0, hasPrev: false, hasNext: false });
      tbody.innerHTML = `<tr><td colspan="9"><span class="pill no">Failed to load alerts</span></td></tr>`;
      if (alertBody && alertBody.classList.contains("page-helmet-alerts")) {
        alertBody.classList.remove("is-hydrating");
      }
      return;
    }

    const alerts = Array.isArray(payload.alerts) ? payload.alerts : [];
    const serverTotalPages = Number(payload.total_pages || 0);
    if (serverTotalPages > 0 && alertPage > serverTotalPages) {
      alertPage = serverTotalPages;
      await populateAlerts();
      return;
    }
    syncAlertPaginationUi({
      total: Number(payload.total || alerts.length),
      page: Number(payload.page || alertPage),
      pageSize: Number(payload.page_size || alertPageSize),
      totalPages: serverTotalPages,
      hasPrev: Boolean(payload.has_prev),
      hasNext: Boolean(payload.has_next),
    });
    const lengthHint = document.getElementById("alert-length-hint");
    if (lengthHint) lengthHint.textContent = `Alerts: ${Number(payload.total || alerts.length)}`;

    if (alerts.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9"><span class="pill">No alerts found</span></td></tr>`;
      if (alertBody && alertBody.classList.contains("page-helmet-alerts")) {
        alertBody.classList.remove("is-hydrating");
      }
      return;
    }

    tbody.innerHTML = alerts
      .map((alert, index) => {
        const uid = String(alert.alert_uid || "");
        const status = String(alert.status || "PENDING");
        const alertName = displayAlertName(alert);
        const camera = displayCamera(alert);
        const date = String(alert.date || "-");
        const time = formatHmsFromIso(alert.start_time_iso) || "-";
        const reviewedTime = ["CONFIRMED", "DISMISSED"].includes(status.toUpperCase()) && alert.review_updated_at_utc
          ? formatDateTimeFromIso(alert.review_updated_at_utc)
          : "-";
        const count = Number(alert.person_count || 0);
        const thumbUrl = alert.thumbnail_url ? String(alert.thumbnail_url) : "";
        const active = index === 0 ? " queue-row-active" : "";
        const linkActive = index === 0 ? " active" : "";
        if (index === 0 && alertOpenSelected instanceof HTMLAnchorElement) {
          alertOpenSelected.href = buildAlertDetailHref(uid);
        }
        return `
          <tr class="${active.trim()}" data-alert-uid="${escapeHtml(uid)}">
            <td><a class="queue-session-link alert-name${linkActive}" href="#${encodeURIComponent(uid)}" aria-label="Open ${escapeHtml(alertName)} alert" data-alert-link>${escapeHtml(alertName)}</a></td>
            <td>${escapeHtml(date)}</td>
            <td>${escapeHtml(time)}</td>
            <td>${escapeHtml(reviewedTime)}</td>
            <td><span class="alert-camera" title="${escapeHtml(camera)}">${escapeHtml(camera)}</span></td>
            <td><span class="pill ink">${Number.isFinite(count) ? count : 0}</span></td>
            <td><span class="pill ${pillClassForAlertStatus(status)}">${displayAlertStatus(status)}</span></td>
            <td>
              <div class="queue-evidence-cell">
                <div class="queue-evidence-preview ${thumbUrl ? "" : "queue-evidence-empty"}">
                  ${thumbUrl ? `<img class="queue-evidence-thumb" alt="alert thumbnail" src="${thumbUrl}" loading="lazy" />` : `<span class="queue-evidence-none">No preview</span>`}
                </div>
              </div>
            </td>
            <td><a class="btn btn-compact action-inspect" href="${buildAlertDetailHref(uid)}">Inspect</a></td>
          </tr>
        `;
      })
      .join("");

    const alertLinks = Array.from(document.querySelectorAll("[data-alert-link]"));
    const setActiveAlert = (index) => {
      alertLinks.forEach((link, linkIndex) => {
        const isActive = linkIndex === index;
        link.classList.toggle("active", isActive);
        const row = link.closest("tr");
        if (row) row.classList.toggle("queue-row-active", isActive);
        if (isActive && alertOpenSelected instanceof HTMLAnchorElement) {
          const uid = row instanceof HTMLTableRowElement ? String(row.dataset.alertUid || "") : "";
          alertOpenSelected.href = buildAlertDetailHref(uid);
        }
      });
    };
    alertLinks.forEach((link, index) => {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        setActiveAlert(index);
        updateHashFromLink(link);
      });
    });
    document.querySelectorAll(".page-helmet-alerts .queue-table tbody tr").forEach((row) => {
      if (!(row instanceof HTMLTableRowElement)) return;
      row.addEventListener("click", (event) => {
        const target = event.target;
        if (target instanceof Element && target.closest("a, button, input, select, textarea, label")) return;
        const link = row.querySelector("[data-alert-link]");
        if (!(link instanceof HTMLAnchorElement)) return;
        const idx = alertLinks.indexOf(link);
        if (idx >= 0) {
          setActiveAlert(idx);
          updateHashFromLink(link);
        }
      });
    });

    if (alertBody && alertBody.classList.contains("page-helmet-alerts")) {
      alertBody.classList.remove("is-hydrating");
    }
  };

  const populateAlertDetail = async () => {
    const alertUid = readAlertUidFromUrl();
    if (!alertUid) {
      try {
        const list = await apiFetchJson(withDateApiQuery("/api/alerts?status=PENDING&page=1&page_size=1&sort=NEWEST"));
        const alerts = Array.isArray(list.alerts) ? list.alerts : [];
        if (alerts.length > 0 && alerts[0].alert_uid) {
          window.location.assign(buildAlertDetailHref(String(alerts[0].alert_uid)));
          return;
        }
      } catch (err) {
        // ignore
      }
      const stage = document.getElementById("alert-thumbnail-stage");
      if (stage instanceof HTMLElement) {
        stage.innerHTML = '<div class="frame-overlay"><span class="pill">tidak ada alert</span></div>';
      }
      return;
    }

    let payload;
    try {
      payload = await apiFetchJson(`/api/alerts/${encodeURIComponent(alertUid)}`);
    } catch (err) {
      return;
    }

    const alert = payload.alert && typeof payload.alert === "object" ? payload.alert : {};
    const nameNode = document.getElementById("alert-detail-name");
    const uidNode = document.getElementById("alert-detail-uid");
    const idNode = document.getElementById("alert-detail-id");
    const dateHint = document.getElementById("alert-detail-date-hint");
    const cameraNode = document.getElementById("alert-detail-camera");
    const areaNode = document.getElementById("alert-detail-area");
    const sessionNode = document.getElementById("alert-detail-session");
    const personPill = document.getElementById("alert-person-count");
    const reviewPill = document.getElementById("alert-review-status");
    const noteBox = document.getElementById("alert-review-note");
    const statusInput = document.getElementById("alert-review-status-input");
    const stage = document.getElementById("alert-thumbnail-stage");
    const backLink = document.getElementById("alert-back-link");
    const listLink = document.getElementById("alert-list-link");

    const alertName = displayAlertName(alert);
    if (nameNode) nameNode.textContent = alertName;
    if (uidNode) uidNode.textContent = displayAlertReference(alertUid);
    if (idNode) idNode.textContent = alertName;
    if (dateHint) dateHint.textContent = `Shift date: ${payload.date || "-"} · ${shiftLabel(payload.shift_id, payload.shift_name)} · Actual time: ${formatDateTimeFromIso(alert.start_time_iso || alert.end_time_iso)} WIB`;
    if (cameraNode) cameraNode.textContent = displayCamera(alert);
    if (areaNode) areaNode.textContent = String(alert.safety_area_id || "-");
    const related = alert.related_session_uid ? String(alert.related_session_uid) : "-";
    if (sessionNode) {
      sessionNode.innerHTML = related !== "-" ? `<a href="${buildSessionDetailHref(related)}">${escapeHtml(related)}</a>` : "-";
    }
    if (personPill) personPill.textContent = `People ${Number(alert.person_count || 0)}`;
    const status = String(payload.status || "PENDING");
    if (reviewPill) {
      reviewPill.className = `pill ${pillClassForAlertStatus(status)}`;
      reviewPill.textContent = `Review ${displayAlertStatus(status)}`;
    }
    if (noteBox instanceof HTMLTextAreaElement) {
      noteBox.value = payload.review && payload.review.review_note ? String(payload.review.review_note) : "";
    }
    if (statusInput instanceof HTMLInputElement) statusInput.value = status;
    if (backLink instanceof HTMLAnchorElement) backLink.href = buildUiHrefWithDate("helmet-alerts.html", encodeURIComponent(alertUid));
    if (listLink instanceof HTMLAnchorElement) listLink.href = buildUiHrefWithDate("helmet-alerts.html", encodeURIComponent(alertUid));

    if (stage instanceof HTMLElement) {
      if (payload.thumbnail_url) {
        stage.innerHTML = `<img alt="alert thumbnail" src="${escapeHtml(payload.thumbnail_url)}" style="width:100%;height:100%;object-fit:cover;border-radius:inherit;" />`;
      } else {
        stage.innerHTML = '<div class="frame-overlay"><span class="pill">tanpa thumbnail</span></div>';
      }
    }

    const alertNavigation = createPendingReviewNavigation({
      currentUid: alertUid, itemsKey: "alerts", uidKey: "alert_uid",
      fetchPage: (page) => apiFetchJson(withDateApiQuery(withReviewFilters(
        `/api/alerts?status=PENDING&page=${page}&page_size=200&sort=NEWEST`, "alert"
      ))),
    });
    const resolveNextPendingAlert = async (refresh = false) => {
      try {
        return await alertNavigation.next({ refresh });
      } catch (err) {
        return null;
      }
    };

    const nextAlertUid = await resolveNextPendingAlert();
    const nextLink = document.getElementById("alert-next-link");
    if (nextLink instanceof HTMLAnchorElement) {
      const nextUid = nextAlertUid;
      if (nextUid && nextUid !== alertUid) {
        nextLink.href = buildAlertDetailHref(nextUid);
        nextLink.textContent = "Next Pending";
      } else {
        nextLink.href = buildUiHrefWithDate("helmet-alerts.html");
        nextLink.textContent = "No Pending";
      }
    }

    const form = document.getElementById("alert-review-form");
    if (form instanceof HTMLFormElement) {
      form.querySelectorAll("button[data-alert-status]").forEach((button) => {
        button.addEventListener("click", () => {
          if (statusInput instanceof HTMLInputElement) {
            statusInput.value = String(button.getAttribute("data-alert-status") || "PENDING");
          }
        });
      });
      form.onsubmit = async (event) => {
        event.preventDefault();
        const nextStatus = statusInput instanceof HTMLInputElement ? String(statusInput.value || "PENDING") : "PENDING";
        const note = noteBox instanceof HTMLTextAreaElement ? String(noteBox.value || "") : "";
        try {
          await apiFetchJson(`/api/alerts/${encodeURIComponent(alertUid)}/review`, {
            method: "PUT",
            body: JSON.stringify({ status: nextStatus, review_note: note }),
          });
          if (String(nextStatus).toUpperCase() !== "PENDING") {
            const nextUid = await resolveNextPendingAlert(true);
            if (nextUid && nextUid !== alertUid) {
              window.location.assign(buildAlertDetailHref(nextUid));
              return;
            }
            window.location.assign(buildUiHrefWithDate("helmet-alerts.html"));
          } else if (reviewPill) {
            reviewPill.className = `pill ${pillClassForAlertStatus(nextStatus)}`;
            reviewPill.textContent = `Review ${displayAlertStatus(nextStatus)}`;
          }
        } catch (err) {
          alert("Failed to save alert review");
        }
      };
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
      const actualTime = startIso ? `${formatDateTimeFromIso(startIso)} WIB` : "unavailable";
      dateHint.textContent = `Shift date: ${sessionDate} · Actual time: ${actualTime}`;
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
    if (overviewDate) overviewDate.textContent = sessionDate;
    const overviewActualTime = document.getElementById("detail-overview-actual-time");
    if (overviewActualTime) overviewActualTime.textContent = startIso ? `Mulai: ${formatDateTimeFromIso(startIso)} WIB` : "Waktu mulai tidak tersedia";
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

    const sessionNavigation = createPendingReviewNavigation({
      currentUid: sessionUid, itemsKey: "sessions", uidKey: "session_uid",
      fetchPage: (page) => apiFetchJson(withDateApiQuery(withReviewFilters(
        `/api/sessions?operator_verdict=NEEDS_REVIEW&reviewable_only=true&sort=NEWEST&page=${page}&page_size=200`, "session"
      ))),
    });
    const resolveNextPending = async (refresh = false) => {
      try {
        return await sessionNavigation.next({ refresh });
      } catch (err) {
        return null;
      }
    };

    const nextSessionUid = await resolveNextPending();
    const nextLink = document.getElementById("detail-next-link");
    if (nextLink instanceof HTMLAnchorElement) {
      const nextUid = nextSessionUid;
      if (nextUid && nextUid !== sessionUid) {
        nextLink.href = buildSessionDetailHref(nextUid);
        nextLink.textContent = "Sesi berikutnya";
      } else {
        nextLink.href = buildUiHrefWithDate("review-queue.html");
        nextLink.textContent = "Tidak ada sesi lain";
      }
    }

    const machine = sopStatusValue(payload, "machine");
    const review = String(payload.review_status || (payload.review && payload.review.review_status) || "PENDING");
    const verdict = operatorVerdict(payload);
    const machinePill = document.getElementById("detail-machine-status");
    const reviewPill = document.getElementById("detail-review-status");
    const finalPill = document.getElementById("detail-final-status");
    if (overviewAi) {
      overviewAi.textContent = machine === "UNKNOWN" ? "Bukti belum cukup" : verdictLabel(machine);
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
      machinePill.textContent = `Hasil AI: ${machine === "UNKNOWN" ? "Bukti belum cukup" : verdictLabel(machine)}`;
    }
    if (reviewPill) {
      reviewPill.className = `pill ${pillClassForReviewStatus(review)}`;
      reviewPill.textContent = reviewSourceLabel(payload.review_source);
    }
    if (finalPill) {
      finalPill.className = `pill ${verdictClass(verdict)}`;
      finalPill.textContent = `Hasil akhir: ${verdictLabel(verdict)}`;
    }

    renderSopPanel(payload);

    const reasonSelect = document.getElementById("scope-reason");
    const reasonField = document.getElementById("scope-reason-field");
    const saveButton = document.getElementById("review-save");
    const reviewFeedback = document.getElementById("detail-review-error");
    const readOnly = Boolean(payload.sop && payload.sop.read_only);
    const syncReviewDecision = () => {
      const excluded = statusInput instanceof HTMLSelectElement && statusInput.value === "OUT_OF_SCOPE";
      if (reasonField) reasonField.hidden = !excluded;
      if (reasonSelect instanceof HTMLSelectElement) {
        reasonSelect.disabled = readOnly || !excluded;
        reasonSelect.required = excluded;
      }
      const sopPanel = document.getElementById("detail-sop-panel");
      if (sopPanel) sopPanel.hidden = excluded;
      const note = document.getElementById("review-note");
      if (note instanceof HTMLTextAreaElement) {
        note.disabled = readOnly;
        note.required = excluded && reasonSelect instanceof HTMLSelectElement && reasonSelect.value === "OTHER";
      }
      if (reviewFeedback) {
        reviewFeedback.className = "caption detail-review-feedback";
        reviewFeedback.textContent = readOnly
          ? "Arsip sesi operator hanya dapat dibaca; keputusan sebelumnya tidak dapat diubah."
          : excluded
          ? "Sesi akan diselesaikan dan tidak dihitung dalam kepatuhan SOP."
          : "Periksa bukti dan koreksi hasil SOP sebelum menyimpan keputusan.";
      }
    };
    if (statusInput instanceof HTMLSelectElement) {
      const scopeOption = statusInput.querySelector('option[value="OUT_OF_SCOPE"]');
      if (scopeOption) scopeOption.disabled = !(payload.sop && payload.sop.profile === "roll_sop_v1");
      statusInput.disabled = readOnly;
      statusInput.value = ["QUALIFIED", "NOT_QUALIFIED", "OUT_OF_SCOPE"].includes(review) ? review : "";
      statusInput.onchange = syncReviewDecision;
    }
    if (reasonSelect instanceof HTMLSelectElement) {
      reasonSelect.value = String(payload.scope_reason || "");
      reasonSelect.onchange = syncReviewDecision;
    }
    if (saveButton instanceof HTMLButtonElement) saveButton.disabled = readOnly;
    syncReviewDecision();

    const noteBox = document.getElementById("review-note");
    if (noteBox instanceof HTMLTextAreaElement) {
      noteBox.value = payload.review && payload.review.review_note ? String(payload.review.review_note) : "";
    }

    const player = document.querySelector(".detail-player");
    const evidenceStrip = document.getElementById("detail-evidence-strip");
    const evidenceState = document.getElementById("detail-evidence-state");
    const evidencePill = document.getElementById("detail-evidence-pill");
    if (player instanceof HTMLElement) {
      mountSessionEvidence({ payload, startS, player, evidenceStrip, evidenceState, evidencePill });
    }

    if (form) {
      form.onsubmit = async (event) => {
        event.preventDefault();
        if (form.classList.contains("is-submitting")) return;
        const reviewError = document.getElementById("detail-review-error");
        if (reviewError) {
          reviewError.className = "caption detail-review-feedback";
          reviewError.textContent = "Menyimpan keputusan...";
        }
        const reviewStatus = statusInput instanceof HTMLSelectElement ? statusInput.value : "";
        if (!reviewStatus) return;
        form.classList.add("is-submitting");
        if (saveButton instanceof HTMLButtonElement) saveButton.disabled = true;
        const note = noteBox instanceof HTMLTextAreaElement ? noteBox.value : "";
        const overrides = {};
        form.querySelectorAll("select[data-override-key]").forEach((node) => {
          if (!(node instanceof HTMLSelectElement)) return;
          const key = node.getAttribute("data-override-key") || "";
          const value = String(node.value || "").trim();
          if (key && value) overrides[key] = value;
        });
        const body = { review_status: reviewStatus, review_note: note };
        if (reviewStatus === "OUT_OF_SCOPE" && reasonSelect instanceof HTMLSelectElement) body.scope_reason = reasonSelect.value;
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
            reviewPill.textContent = "Ditinjau petugas";
          }
          if (String(reviewStatus || "").toUpperCase() !== "PENDING") {
            const nextUid = await resolveNextPending(true);
            if (nextUid && nextUid !== sessionUid) {
              window.location.assign(buildSessionDetailHref(nextUid));
              return;
            }
            window.location.assign(buildUiHrefWithDate("review-queue.html"));
          }
        } catch (err) {
          form.classList.remove("is-submitting");
          if (saveButton instanceof HTMLButtonElement) saveButton.disabled = readOnly;
          if (reviewError) {
            const errorText = String(err);
            let apiDetail = "";
            const jsonStart = errorText.indexOf("{");
            if (jsonStart >= 0) {
              try {
                const responseBody = JSON.parse(errorText.slice(jsonStart));
                apiDetail = typeof responseBody.detail === "string" ? responseBody.detail : "";
              } catch (_) {
                apiDetail = "";
              }
            }
            reviewError.className = "validation-summary no detail-review-feedback";
            reviewError.textContent = errorText.includes("Review decision requires final SOP result")
              ? "Belum disimpan: hasil SOP akhir belum sesuai keputusan ini. Periksa bukti, lalu koreksi hasil langkah atau hasil akhir sebelum mencoba lagi."
              : (errorText.includes("HTTP 400") || errorText.includes("HTTP 422"))
              ? `Belum disimpan: ${apiDetail || "periksa isian review, lalu coba lagi."}`
              : "Keputusan belum tersimpan. Coba lagi.";
          }
        }
      };
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

  let dashboardLoadVersion = 0;
  const populateDashboard = async () => {
    const loadVersion = ++dashboardLoadVersion;
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
    if (dashboardDateLabel) dashboardDateLabel.textContent = dashboardDateRange();
    renderDashboardSummary(null, "loading");
    setDashboardChartState("loading", "Memuat grafik…");
    const chartTable = document.getElementById("trend-data-body");
    if (chartTable) chartTable.innerHTML = "";
    const latestTable = document.getElementById("dashboard-latest-sessions-body");
    if (latestTable) latestTable.innerHTML = '<tr><td colspan="6">Memuat sesi…</td></tr>';
    let statsLoaded = false;
    try {
      const s = await apiFetchJson(withDateApiQuery("/api/stats"));
      if (loadVersion !== dashboardLoadVersion) return;
      renderDashboardSummary(s);
      statsLoaded = true;

      const renderDashboardTrend = async () => {
        const svg = document.querySelector(".trend-svg");
        if (!(svg instanceof SVGSVGElement)) return;

        const trendSlice = document.getElementById("trend-active-slice");
        const trendXTitle = document.getElementById("trend-x-title");

        const donePath = svg.querySelector("path.trend-line.yes");
        const unknownPath = svg.querySelector("path.trend-line.pending");
        const notDonePath = svg.querySelector("path.trend-line.no");
        const excludedPath = svg.querySelector("path.trend-line.excluded");
        const doneDot = svg.querySelector("circle.trend-focus-dot.yes");
        const unknownDot = svg.querySelector("circle.trend-focus-dot.pending");
        const notDoneDot = svg.querySelector("circle.trend-focus-dot.no");
        const excludedDot = svg.querySelector("circle.trend-focus-dot.excluded");
        const gridValues = Array.from(svg.querySelectorAll("text.trend-grid-value"));
        const gridLines = Array.from(svg.querySelectorAll("line.trend-grid-line"));
        const axisLabels = Array.from(svg.querySelectorAll("text.trend-axis-label"));

        if (
          !(donePath instanceof SVGPathElement) ||
          !(unknownPath instanceof SVGPathElement) ||
          !(notDonePath instanceof SVGPathElement) ||
          !(excludedPath instanceof SVGPathElement)
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
          if (loadVersion === dashboardLoadVersion) setDashboardChartState("error", "Grafik gagal dimuat. Terapkan kembali tanggal untuk mencoba lagi.");
          return;
        }
        if (loadVersion !== dashboardLoadVersion) return;

        const sessions = Array.isArray(listPayload.sessions) ? listPayload.sessions : [];

        const useHourly =
          (slice.from && slice.to && slice.from === slice.to) ||
          (!slice.from && !slice.to && typeof listUrl === "string" && listUrl.includes("date="));

        if (trendXTitle instanceof HTMLElement) {
          trendXTitle.textContent = useHourly ? "Jam (WIB), mulai 07:30" : "Tanggal shift (WIB)";
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
            trendSlice.textContent = activeDate ? `${slice.from && slice.to ? "Jumlah sesi per jam" : "Hari terbaru, per jam"} · ${displayShiftDate(activeDate)}` : "Jumlah sesi per jam · tanggal terbaru";
          } else {
            // Keep this aligned with the rest of the dashboard date filter.
            trendSlice.textContent = `Jumlah sesi per hari · ${dashboardDateRange()}`;
          }
        }

        let labels = [];
        let done = [];
        let unknown = [];
        let notDone = [];
        let excluded = [];

        if (useHourly) {
          // Hourly windows run from 07:30 today through 07:30 the next day.
          labels = Array.from({ length: 24 }, (_, h) => `${String((h + 7) % 24).padStart(2, "0")}:30`);
          done = Array(24).fill(0);
          unknown = Array(24).fill(0);
          notDone = Array(24).fill(0);
          excluded = Array(24).fill(0);
          sessions.forEach((row) => {
            const iso =
              row && row.start_time_iso ? String(row.start_time_iso) : row && row.end_time_iso ? String(row.end_time_iso) : "";
            if (!iso) return;
            const idx = shiftDayHourIndex(iso, String(row.shift_date || row.date || ""));
            if (idx < 0 || idx >= 24) return;
            const status = dashboardTrendStatus(row);
            if (status === "DONE") done[idx] += 1;
            else if (status === "NOT_DONE") notDone[idx] += 1;
            else if (status === "NEEDS_REVIEW") unknown[idx] += 1;
            else if (status === "OUT_OF_SCOPE") excluded[idx] += 1;
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
          excluded = Array(labels.length).fill(0);
          const idxByDate = new Map(labels.map((d, i) => [d, i]));

          sessions.forEach((row) => {
            const d = row && row.date ? String(row.date) : "";
            const idx = idxByDate.get(d);
            if (idx == null) return;
            const status = dashboardTrendStatus(row);
            if (status === "DONE") done[idx] += 1;
            else if (status === "NOT_DONE") notDone[idx] += 1;
            else if (status === "NEEDS_REVIEW") unknown[idx] += 1;
            else if (status === "OUT_OF_SCOPE") excluded[idx] += 1;
          });
        }

        // Integer ticks and their grid lines use the same scale as the data.
        const maxVal = Math.max(3, Math.ceil(Math.max(0, ...done, ...unknown, ...notDone, ...excluded) / 3) * 3);
        const xMin = 64;
        const xMax = 888;
        const yTop = 32;
        const yBottom = 232;
        const binCount = labels.length;
        const xStep = binCount > 1 ? (xMax - xMin) / (binCount - 1) : 0;
        const xAt = (index) => binCount === 1 ? (xMin + xMax) / 2 : xMin + index * xStep;
        const yScale = (value) => yBottom - (Math.max(0, Number(value || 0)) / maxVal) * (yBottom - yTop);
        for (const [values, line, dot] of [
          [done, donePath, doneDot], [unknown, unknownPath, unknownDot],
          [notDone, notDonePath, notDoneDot], [excluded, excludedPath, excludedDot],
        ]) {
          const points = values.map((value, index) => ({ x: xAt(index), y: yScale(value) }));
          // Straight segments cannot invent negative counts between samples.
          line.setAttribute("d", buildCountPath(points));
          if (dot instanceof SVGCircleElement && points.length) {
            const last = points[points.length - 1];
            dot.setAttribute("cx", String(last.x.toFixed(1)));
            dot.setAttribute("cy", String(last.y.toFixed(1)));
          }
        }
        [maxVal, maxVal * 2 / 3, maxVal / 3, 0].forEach((value, index) => {
          const y = yScale(value);
          if (gridValues[index]) {
            gridValues[index].textContent = countFormat.format(value);
            gridValues[index].setAttribute("y", String(y + 4));
          }
          if (gridLines[index]) {
            gridLines[index].setAttribute("y1", String(y));
            gridLines[index].setAttribute("y2", String(y));
          }
        });
        const tickIndices = useHourly ? [0, 6, 12, 18, 23] : [...new Set([
          0, Math.floor((binCount - 1) * .25), Math.floor((binCount - 1) * .5),
          Math.floor((binCount - 1) * .75), binCount - 1,
        ])];
        axisLabels.forEach((node, index) => {
          const tick = tickIndices[index];
          node.textContent = tick == null ? "" : useHourly ? labels[tick] : displayShiftDate(labels[tick], true);
          if (tick != null) node.setAttribute("x", String(xAt(tick)));
        });
        if (chartTable) chartTable.innerHTML = labels.map((label, index) => `
          <tr><th scope="row">${escapeHtml(useHourly ? label : displayShiftDate(label))}</th>
          <td>${countFormat.format(unknown[index])}</td><td>${countFormat.format(done[index])}</td>
          <td>${countFormat.format(notDone[index])}</td><td>${countFormat.format(excluded[index])}</td></tr>
        `).join("");
        const plotted = [...done, ...unknown, ...notDone, ...excluded].reduce((total, value) => total + value, 0);
        const skipped = sessions.length - plotted;
        setDashboardChartState(plotted ? "ready" : "empty", skipped > 0 ?
          `${countFormat.format(skipped)} sesi tidak memiliki waktu yang dapat ditampilkan pada grafik.` :
          plotted ? `${countFormat.format(plotted)} sesi dalam grafik` : "Tidak ada sesi pada periode grafik.");
      };

      await renderDashboardTrend();
      if (loadVersion !== dashboardLoadVersion) return;

      const latestBody = document.getElementById("dashboard-latest-sessions-body");
      if (latestBody instanceof HTMLTableSectionElement) {
        try {
          const latestPayload = await apiFetchJson(withDateApiQuery("/api/sessions?limit=3"));
          if (loadVersion !== dashboardLoadVersion) return;
          const latestSessions = Array.isArray(latestPayload.sessions) ? latestPayload.sessions : [];
          if (latestSessions.length === 0) {
            latestBody.innerHTML = `<tr><td colspan="6"><span class="pill">No sessions found</span></td></tr>`;
          } else {
            latestBody.innerHTML = latestSessions
              .map((session) => {
                const uid = String(session.session_uid || "");
                const sid = String(session.session_id || uid || "-");
                const start = formatHmsFromIso(session.start_time_iso);
                const source = String(session.review_source || "PENDING");
                const remark =
                  Number(session.clip_count || 0) > 0
                    ? `${countFormat.format(Number(session.clip_count))} klip bukti`
                    : session.has_thumbnail
                    ? "Thumbnail tersedia"
                    : "Belum ada bukti";
                return `
                  <tr>
                    <td><strong>${escapeHtml(sid)}</strong></td>
                    <td>${start}</td>
                    <td>${dashboardSopPills(session)}</td>
                    <td>${escapeHtml(reviewSourceLabel(source))}</td>
                    <td>${escapeHtml(remark)}</td>
                    <td><a class="btn btn-compact action-inspect" href="${buildSessionDetailHref(uid)}">Lihat sesi</a></td>
                  </tr>
                `;
              })
              .join("");
          }
        } catch (err) {
          if (loadVersion !== dashboardLoadVersion) return;
          latestBody.innerHTML = `<tr><td colspan="6"><span class="pill no">Failed to load sessions</span></td></tr>`;
        }
      }
    } catch (err) {
      if (!statsLoaded && loadVersion === dashboardLoadVersion) {
        renderDashboardSummary(null, "error");
        setDashboardChartState("error", "Grafik belum tersedia. Muat ringkasan kembali.");
        if (latestTable) latestTable.innerHTML = '<tr><td colspan="6">Sesi belum tersedia</td></tr>';
      }
    } finally {
      if (loadVersion === dashboardLoadVersion && dashboardBody && dashboardBody.classList.contains("page-dashboard")) {
        dashboardBody.classList.remove("is-hydrating");
      }
    }
  };

  const restoreQueueContext = (kind, controls) => {
    const context = readReviewContext(kind);
    for (const [key, id] of Object.entries(controls)) {
      const node = document.getElementById(id);
      if (node instanceof HTMLSelectElement && Array.from(node.options).some((option) => option.value === context[key])) {
        node.value = context[key];
      }
    }
    const page = Number(context.page);
    return Number.isSafeInteger(page) && page > 0 ? page : 1;
  };

  const body = document.body;
  applyDateSliceToStaticNav();
  if (body && body.classList.contains("page-review-queue")) {
    document.getElementById("queue-load-retry")?.addEventListener("click", () => populateQueue());
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
    queuePage = restoreQueueContext("session", {
      verdict: "queue-status", evidence: "queue-evidence", shift: "queue-shift",
      sort: "queue-sort", page_size: "queue-page-size",
    });
    const statusSel = document.getElementById("queue-status");
    const requestedVerdict = new URLSearchParams(window.location.search).get("verdict");
    if (statusSel instanceof HTMLSelectElement && ["DONE", "NOT_DONE", "NEEDS_REVIEW", "OUT_OF_SCOPE"].includes(requestedVerdict)) statusSel.value = requestedVerdict;
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
  if (body && body.classList.contains("page-helmet-alerts")) {
    bindStaleIndicator({ pillId: "stale-pill-alerts", onRefresh: () => populateAlerts() });
    bindDateControls({
      fromId: "alert-date-from",
      toId: "alert-date-to",
      applyId: "alert-date-apply",
      clearId: "alert-date-clear",
      labelId: "alert-active-date-slice",
      onChange: () => {
        resetAlertPage();
        populateAlerts();
      },
    });
    alertPage = restoreQueueContext("alert", {
      status: "alert-status", sort: "alert-sort", page_size: "alert-page-size",
    });
    const alertStatusSel = document.getElementById("alert-status");
    const alertSortSel = document.getElementById("alert-sort");
    const alertPageSizeSel = document.getElementById("alert-page-size");
    [alertStatusSel, alertSortSel, alertPageSizeSel].forEach((node) => {
      if (node instanceof HTMLSelectElement) {
        node.addEventListener("change", () => {
          resetAlertPage();
          populateAlerts();
        });
      }
    });
    if (alertPagePrevBtn instanceof HTMLButtonElement) {
      alertPagePrevBtn.addEventListener("click", () => {
        if (alertPage <= 1) return;
        alertPage -= 1;
        populateAlerts();
      });
    }
    if (alertPageNextBtn instanceof HTMLButtonElement) {
      alertPageNextBtn.addEventListener("click", () => {
        alertPage += 1;
        populateAlerts();
      });
    }
    alertPageSize = readAlertPageSize();
    populateAlerts();
  }
  if (body && body.classList.contains("page-helmet-alert-detail")) {
    window.addEventListener("hashchange", () => {
      populateAlertDetail();
    });
    populateAlertDetail();
  }
  initSetupPage();
  if (body && body.classList.contains("page-dashboard")) {
    document.getElementById("dashboard-summary-retry")?.addEventListener("click", () => populateDashboard());
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
