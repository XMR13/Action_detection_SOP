// Date filters and navigation shared by the review pages.
// Importing this module does not bind events; site.js calls the functions below.
// Every date filter selects a shift-start date, rather than a calendar day.

import { copyReviewContext } from "./review-navigation.js";

const DATE_YMD_RE = /^\d{4}-\d{2}-\d{2}$/;

const isValidDateYmd = (raw) => DATE_YMD_RE.test(String(raw || ""));

export const currentShiftDate = (now = new Date()) => {
  // Read the clock in WIB, then move its day boundary from midnight to 07:30.
  // UTC getters/formatting here deliberately avoid the browser's local timezone.
  const wibOffsetMs = 7 * 60 * 60 * 1000;
  const shiftDayStartMs = (7 * 60 + 30) * 60 * 1000;
  return new Date(now.getTime() + wibOffsetMs - shiftDayStartMs).toISOString().slice(0, 10);
};

export const shiftDateDaysAgo = (days, now = new Date()) => {
  const day = new Date(`${currentShiftDate(now)}T00:00:00Z`);
  day.setUTCDate(day.getUTCDate() - days);
  return day.toISOString().slice(0, 10);
};

export const toYmdLocal = (d) => {
    //just return the from number to string
  const yy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yy}-${mm}-${dd}`;
};

export const readDateSliceFromUrl = () => {
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

export const dateSliceLabel = () => {
  const slice = readDateSliceFromUrl();
  if (slice.from && slice.to) {
    if (slice.from === slice.to) return `Shift date: ${slice.from}`;
    return `Shift date: ${slice.from} -> ${slice.to}`;
  }
  if (slice.from) return `Shift date: from ${slice.from}`;
  if (slice.to) return `Shift date: until ${slice.to}`;
  return "Shift date: all";
};

const buildDateApiQuery = () => {
  const slice = readDateSliceFromUrl();
  const params = new URLSearchParams();
  if (slice.from) params.set("date_from", slice.from);
  if (slice.to) params.set("date_to", slice.to);
  return params.toString();
};

export const withDateApiQuery = (url) => {
  const query = buildDateApiQuery();
  if (!query) return url;
  return `${url}${url.includes("?") ? "&" : "?"}${query}`;
};

export const buildUiHrefWithDate = (page, hashRaw) => {
  const slice = readDateSliceFromUrl();
  const params = new URLSearchParams();
  if (slice.from) params.set("date_from", slice.from);
  if (slice.to) params.set("date_to", slice.to);
  if (["review-queue.html", "session-detail.html"].includes(page)) copyReviewContext(params, "session");
  if (["helmet-alerts.html", "helmet-alert-detail.html"].includes(page)) copyReviewContext(params, "alert");
  const query = params.toString();
  const hash = hashRaw ? `#${hashRaw}` : "";
  return `${page}${query ? `?${query}` : ""}${hash}`;
};

export const buildSessionDetailHref = (sessionUid) => {
  const slice = readDateSliceFromUrl();
  const params = new URLSearchParams();
  if (slice.from) params.set("date_from", slice.from);
  if (slice.to) params.set("date_to", slice.to);
  copyReviewContext(params, "session");
  if (sessionUid) params.set("session_uid", String(sessionUid));
  const query = params.toString();
  const hash = sessionUid ? `#${encodeURIComponent(String(sessionUid))}` : "";
  return `session-detail.html${query ? `?${query}` : ""}${hash}`;
};

export const buildAlertDetailHref = (alertUid) => {
  const slice = readDateSliceFromUrl();
  const params = new URLSearchParams();
  if (slice.from) params.set("date_from", slice.from);
  if (slice.to) params.set("date_to", slice.to);
  copyReviewContext(params, "alert");
  if (alertUid) params.set("alert_uid", String(alertUid));
  const query = params.toString();
  const hash = alertUid ? `#${encodeURIComponent(String(alertUid))}` : "";
  return `helmet-alert-detail.html${query ? `?${query}` : ""}${hash}`;
};

export const readSessionUidFromUrl = () => {
  const params = new URLSearchParams(window.location.search || "");
  const fromQuery = params.get("session_uid");
  if (fromQuery) return String(fromQuery);
  const hash = window.location.hash ? window.location.hash.slice(1) : "";
  return hash ? decodeURIComponent(hash) : "";
};

export const readAlertUidFromUrl = () => {
  const params = new URLSearchParams(window.location.search || "");
  const fromQuery = params.get("alert_uid");
  if (fromQuery) return String(fromQuery);
  const hash = window.location.hash ? window.location.hash.slice(1) : "";
  return hash ? decodeURIComponent(hash) : "";
};

export const applyDateSliceToStaticNav = () => {
  const navTargets = new Set(["index.html", "review-queue.html", "helmet-alerts.html", "helmet-alert-detail.html", "session-detail.html", "setup.html"]);
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

export const bindDateControls = ({
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
      const now = new Date();
      const today = currentShiftDate(now);
      const last7 = shiftDateDaysAgo(6, now);
      const last30 = shiftDateDaysAgo(29, now);
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
      const now = new Date();
      const today = currentShiftDate(now);
      let from = "";
      let to = "";
      const choice = String(rangeSel.value || "CUSTOM");
      if (choice === "TODAY") {
        from = today;
        to = today;
      } else if (choice === "LAST_7_DAYS") {
        from = shiftDateDaysAgo(6, now);
        to = today;
      } else if (choice === "LAST_30_DAYS") {
        from = shiftDateDaysAgo(29, now);
        to = today;
      }
      if (fromInput instanceof HTMLInputElement) fromInput.value = from;
      if (toInput instanceof HTMLInputElement) toInput.value = to;
      if (choice !== "CUSTOM") applyFromInputs();
    });
  }

  refreshUi();
};
