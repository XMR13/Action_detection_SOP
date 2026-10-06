// Historical timestamps without an offset were recorded in WIB.
// Explicit offsets (including UTC Z) are converted to WIB for display.
export const parseWibDateTime = (iso) => {
    const raw = String(iso || "").trim();
    const withOffset = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(raw);
    return new Date(raw && !withOffset ? `${raw}+07:00` : raw);
  };

export const shiftDayHourIndex = (iso, shiftDate) => {
    const d = parseWibDateTime(iso);
    if (Number.isNaN(d.getTime()) || !/^\d{4}-\d{2}-\d{2}$/.test(String(shiftDate || ""))) return -1;
    const dayStart = new Date(`${shiftDate}T07:30:00+07:00`);
    if (Number.isNaN(dayStart.getTime()) || dayStart.toISOString().slice(0, 10) !== shiftDate) return -1;
    // Measure from the assigned day's 07:30 WIB boundary, not just the clock hour.
    const elapsedHours = (d.getTime() - dayStart.getTime()) / (60 * 60 * 1000);
    // A session can start before its assigned day under the largest-overlap rule.
    // Show it at the start of that day instead of wrapping it to the final hour.
    if (elapsedHours < 0) return 0;
    if (elapsedHours >= 24) return -1;
    return Math.floor(elapsedHours);
  };

export const formatHmsFromIso = (iso) => {
    if (!iso) return "-";
    const d = parseWibDateTime(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleTimeString("en-GB", { hour12: false, timeZone: "Asia/Jakarta" });
  };


    export const formatDateTimeFromIso = (iso) => {
    if (!iso) return "-";
    const d = parseWibDateTime(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString("en-GB", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
      timeZone: "Asia/Jakarta",
    });
  };

//format duration from seconds into a goo format
export const formatDuration = (seconds) => {
    const s = Math.max(0, Number(seconds || 0));
    const mm = Math.floor(s / 60);
    const ss = s - mm * 60;
    const mmStr = String(mm).padStart(2, "0");
    const ssStr = ss.toFixed(1).padStart(4, "0");
    return `${mmStr}:${ssStr}`;
  };

export const normalizeShiftId = (raw) => {
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

//get the sfhit ID and return it intto Shift
export const shiftLabel = (shiftId, shiftName) => {
    const explicit = String(shiftName || "").trim();
    if (explicit) return explicit;
    const sid = normalizeShiftId(shiftId);
    if (sid === "S1") return "Shift 1";
    if (sid === "S2") return "Shift 2";
    if (sid === "S3") return "Shift 3";
    return "-";
  };

export const escapeHtml = (raw) =>
    String(raw ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
