(function () {
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
  const queueLinks = Array.from(document.querySelectorAll("[data-queue-link]"));
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
      selectedDetailLink.href = `session-detail.html#${encodeURIComponent(resolvedSessionId.toLowerCase())}`;
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
})();
