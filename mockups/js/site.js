(function () {
  const apiFetchJson = async (path, options) => {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options && options.headers ? options.headers : {}) },
      ...options,
    });
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
    if (v === "QUALIFIED") return "APPROVED";
    if (v === "NOT_QUALIFIED") return "REJECTED";
    return "PENDING";
  };

  const pillClassForReviewStatus = (raw) => {
    const v = String(raw || "PENDING").toUpperCase();
    if (v === "QUALIFIED") return "yes";
    if (v === "NOT_QUALIFIED") return "no";
    return "pending";
  };

  const formatHmsFromIso = (iso) => {
    if (!iso) return "-";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleTimeString("en-GB", { hour12: false });
  };

  const formatDuration = (seconds) => {
    const s = Math.max(0, Number(seconds || 0));
    const mm = Math.floor(s / 60);
    const ss = s - mm * 60;
    const mmStr = String(mm).padStart(2, "0");
    const ssStr = ss.toFixed(1).padStart(4, "0");
    return `${mmStr}:${ssStr}`;
  };

  const lockFormOnSubmit = (form) => {
    form.addEventListener("submit", () => {
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

  const formNode = document.getElementById("review-form");
  const form = formNode instanceof HTMLFormElement ? formNode : null;
  const statusInput = document.getElementById("review-status");
  const actionButtons = form ? form.querySelectorAll("button[data-review-status]") : [];
  let queueLinks = [];
  const selectedSessionIdInput = document.getElementById("selected-session-id");
  const selectedSessionInline = document.getElementById("selected-session-inline");
  const selectedSessionLabel = document.getElementById("selected-session-label");
  const selectedSessionUid = document.getElementById("selected-session-uid");
  const selectedMachineStatus = document.getElementById("selected-machine-status");
  const selectedHumanStatus = document.getElementById("selected-human-status");
  const selectedFinalStatus = document.getElementById("selected-final-status");
  const selectedEvidence = document.getElementById("selected-evidence");
  const selectedSessionSla = document.getElementById("selected-session-sla");
  const selectedDetailLink = document.getElementById("selected-detail-link");

  const syncReviewDock = (link) => {
    const row = link.closest("tr");
    if (!(row instanceof HTMLTableRowElement)) {
      return;
    }

    const { sessionId, sessionUid, shift, machine, human, final, evidence, sla } = row.dataset;
    const resolvedSessionId = sessionId || link.textContent?.trim() || "UNKNOWN";
    const resolvedShift = shift || "-";
    const resolvedMachine = machine || "-";
    const resolvedHuman = human || "-";
    const resolvedFinal = final || "-";
    const resolvedEvidence = evidence || "-";
    const resolvedSessionUid = sessionUid || "-";
    const resolvedSla = sla || "-";

    if (selectedSessionIdInput instanceof HTMLInputElement) {
      selectedSessionIdInput.value = resolvedSessionId;
    }
    if (selectedSessionInline) {
      selectedSessionInline.textContent = resolvedSessionId;
    }
    if (selectedSessionLabel) {
      selectedSessionLabel.textContent = `${resolvedSessionId} | ${resolvedShift} | Machine ${resolvedMachine}`;
    }
    if (selectedSessionUid) {
      selectedSessionUid.textContent = resolvedSessionUid;
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
    if (selectedSessionSla) {
      selectedSessionSla.textContent = resolvedSla === "reviewed" ? "Reviewed" : `SLA ${resolvedSla}`;
    }
    if (selectedDetailLink instanceof HTMLAnchorElement) {
      const targetUid = resolvedSessionUid && resolvedSessionUid !== "-" ? resolvedSessionUid : resolvedSessionId;
      selectedDetailLink.href = `session-detail.html#${encodeURIComponent(String(targetUid))}`;
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
    if (!(tbody instanceof HTMLTableSectionElement)) {
      return;
    }

    let payload;
    try {
      payload = await apiFetchJson("/api/sessions");
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="9"><span class="pill no">Failed to load sessions</span></td></tr>`;
      return;
    }

    const sessions = Array.isArray(payload.sessions) ? payload.sessions : [];
    if (sessions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9"><span class="pill">No sessions found</span></td></tr>`;
      return;
    }

    const rowsHtml = sessions
      .map((s, index) => {
        const uid = String(s.session_uid || "");
        const sid = String(s.session_id || uid || "-");
        const machine = String(s.machine_helmet || "UNKNOWN");
        const review = String(s.review_status || "PENDING");
        const final = String(s.final_helmet || machine);
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
            data-shift="-"
            data-machine="${displayStepStatus(machine)}"
            data-human="${displayReviewStatus(review)}"
            data-final="${displayStepStatus(final)}"
            data-evidence="${evidenceLabel}"
            data-sla="${sla}"
          >
            <td>
              <a class="queue-session-link${linkActive}" href="#${encodeURIComponent(uid)}" data-queue-link>${sid}</a>
              <div class="table-sub">${uid}</div>
            </td>
            <td>${start}</td>
            <td>${dur}</td>
            <td><span class="pill ${pillClassForStepStatus(machine)}">${displayStepStatus(machine)}</span></td>
            <td><span class="pill ${pillClassForReviewStatus(review)}">${displayReviewStatus(review)}</span></td>
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
            <td><a class="btn btn-compact action-inspect" href="session-detail.html#${encodeURIComponent(uid)}">Inspect</a></td>
          </tr>
        `;
      })
      .join("");

    tbody.innerHTML = rowsHtml;
    initQueueInteractions();
  };

  const populateDetail = async () => {
    const hash = window.location.hash ? window.location.hash.slice(1) : "";
    const sessionUid = hash ? decodeURIComponent(hash) : "";
    if (!sessionUid) {
      // If user opens Session Detail from the sidebar, pick the newest session.
      try {
        const list = await apiFetchJson("/api/sessions?limit=1");
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
    const uidHint = document.getElementById("detail-session-uid-hint");
    const sidNode = document.getElementById("detail-session-id");
    if (sidNode) sidNode.textContent = sessionId;
    if (uidHint) uidHint.textContent = `UID: ${sessionUid}`;
    if (selectedSessionIdInput instanceof HTMLInputElement) {
      selectedSessionIdInput.value = sessionId;
    }

    const machine = String(payload.machine_helmet || "UNKNOWN");
    const review = payload.review && payload.review.review_status ? String(payload.review.review_status) : "PENDING";
    const machinePill = document.getElementById("detail-machine-status");
    const reviewPill = document.getElementById("detail-review-status");
    if (machinePill) {
      machinePill.className = `pill ${pillClassForStepStatus(machine)}`;
      machinePill.textContent = `machine ${displayStepStatus(machine)}`;
    }
    if (reviewPill) {
      reviewPill.className = `pill ${pillClassForReviewStatus(review)}`;
      reviewPill.textContent = `review ${displayReviewStatus(review)}`;
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
    if (player instanceof HTMLElement) {
      const renderThumbnail = () => {
        if (payload.thumbnail_url) {
          const url = String(payload.thumbnail_url);
          player.innerHTML = `<img alt="thumbnail" src="${url}" style="width:100%;height:100%;object-fit:cover;border-radius:inherit;" />`;
          return;
        }
        player.innerHTML = '<div class="frame-overlay"><span class="pill">no clip</span></div>';
      };

      const clips = Array.isArray(payload.clips) ? payload.clips : [];
      if (clips.length > 0 && clips[0].url) {
        const clipUrl = String(clips[0].url);
        player.innerHTML =
          '<video id="detail-video" controls preload="metadata" playsinline style="width:100%;height:100%;object-fit:cover;border-radius:inherit;"></video><div class="frame-overlay"><a id="detail-clip-link" class="pill" target="_blank" rel="noreferrer">open clip file</a></div>';
        const video = document.getElementById("detail-video");
        const clipLink = document.getElementById("detail-clip-link");
        if (clipLink instanceof HTMLAnchorElement) {
          clipLink.href = clipUrl;
        }
        if (video instanceof HTMLVideoElement) {
          video.src = clipUrl;
          video.addEventListener(
            "error",
            () => {
              // Browser cannot decode this clip codec (common with OpenCV mp4v).
              renderThumbnail();
            },
            { once: true }
          );
        }
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
    if (!totalNode && !pendingNode) {
      return;
    }
    try {
      const s = await apiFetchJson("/api/stats");
      if (totalNode) totalNode.textContent = String(s.total_sessions ?? "-");
      if (pendingNode) pendingNode.textContent = String(s.pending ?? "-");
    } catch (err) {
      // ignore
    }
  };

  const body = document.body;
  if (body && body.classList.contains("page-review-queue")) {
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
    populateDashboard();
  }
})();
