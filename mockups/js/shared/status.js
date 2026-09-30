// SOP and alert status interpretation shared by the review pages.
// These helpers accept data and return values without fetching or updating the DOM.

export const displayStepStatus = (raw) => {
  const status = String(raw || "").toUpperCase();
  if (status === "DONE") return "Sudah dilakukan";
  if (status === "NOT_DONE") return "Belum dilakukan";
  if (status === "UNKNOWN") return "Bukti belum cukup";
  return "-";
};

export const pillClassForStepStatus = (raw) => {
  const v = String(raw || "").toUpperCase();
  if (v === "DONE") return "yes";
  if (v === "NOT_DONE") return "no";
  if (v === "UNKNOWN") return "dir-b";
  return "";
};

export const operatorVerdict = (row) => {
  const fromApi = String((row && row.operator_verdict) || "").toUpperCase();
  if (["DONE", "NOT_DONE", "NEEDS_REVIEW", "OUT_OF_SCOPE"].includes(fromApi)) return fromApi;
  const review = String((row && row.review_status) || "PENDING").toUpperCase();
  const final = sopStatusValue(row, "final");
  if (review === "OUT_OF_SCOPE") return "OUT_OF_SCOPE";
  if (review === "QUALIFIED" && final === "DONE") return "DONE";
  if (review === "NOT_QUALIFIED" && final === "NOT_DONE") return "NOT_DONE";
  return "NEEDS_REVIEW";
};

const scopeReasonLabel = (reason) => ({ PASSING_THROUGH: "Hanya melintas", ALREADY_WRAPPED: "Sudah dibungkus", OTHER: "Lainnya" }[reason] || "");

export const verdictLabel = (verdict) => {
  if (verdict === "OUT_OF_SCOPE") return "Di luar cakupan SOP";
  if (verdict === "DONE") return "Sesuai SOP";
  if (verdict === "NOT_DONE") return "Tidak sesuai SOP";
  return "Perlu ditinjau";
};

export const verdictClass = (verdict) => {
  if (verdict === "OUT_OF_SCOPE") return "dir-b";
  if (verdict === "DONE") return "yes";
  if (verdict === "NOT_DONE") return "no";
  return "pending";
};

export const reviewSourceLabel = (raw) => {
  const source = String(raw || "PENDING").toUpperCase();
  if (source === "AUTO") return "Otomatis";
  if (source === "MANUAL") return "Ditinjau petugas";
  return "Menunggu tinjauan";
};

export const pillClassForReviewStatus = (raw) => {
  const v = String(raw || "PENDING").toUpperCase();
  if (v === "OUT_OF_SCOPE") return "dir-b";
  if (v === "QUALIFIED") return "yes";
  if (v === "NOT_QUALIFIED") return "no";
  return "pending";
};

export const displayAlertStatus = (raw) => {
  const v = String(raw || "PENDING").toUpperCase();
  if (v === "CONFIRMED") return "CONFIRMED";
  if (v === "DISMISSED") return "DISMISSED";
  return "PENDING";
};

export const pillClassForAlertStatus = (raw) => {
  const v = String(raw || "PENDING").toUpperCase();
  if (v === "CONFIRMED") return "yes";
  if (v === "DISMISSED") return "no";
  return "pending";
};

export const structuredSop = (row) => {
  const sop = row && row.sop && typeof row.sop === "object" ? row.sop : null;
  return sop && sop.profile ? sop : null;
};

export const sopScope = (row, scope) => {
  const sop = structuredSop(row);
  const data = sop && sop[scope] && typeof sop[scope] === "object" ? sop[scope] : null;
  return data || {};
};

export const sopStatusValue = (row, scope) => {
  const data = sopScope(row, scope);
  const fallback =
    scope === "final"
      ? row && (row.final_sop || row.final_helmet || row.machine_sop || row.machine_helmet)
      : row && (row.machine_sop || row.machine_helmet);
  return String(data.status || fallback || "UNKNOWN").toUpperCase();
};

export const dashboardTrendStatus = (row) => {
  return operatorVerdict(row);
};

export const rollOverallDisplay = (raw) => {
  const v = String(raw || "UNKNOWN").toUpperCase();
  if (v === "SESUAI SOP") return "Sesuai SOP";
  if (v === "TIDAK SESUAI SOP") return "Tidak sesuai SOP";
  return "Bukti belum cukup";
};

const queueStepSummary = (row) => {
  const sop = structuredSop(row);
  if (!sop || sop.profile !== "roll_sop_v1") return "";
  const steps = sopScope(row, "final");
  return [
    `Cleaning: ${String(steps.cleaned || "UNKNOWN").toUpperCase()}`,
    `Labeling: ${String(steps.labeled || "UNKNOWN").toUpperCase()}`,
  ].join(" · ");
};

export const queueDecision = (row) => {
  const review = String((row && row.review_status) || "PENDING").toUpperCase();
  const verdict = operatorVerdict(row);
  return {
    label: verdictLabel(verdict),
    className: verdictClass(verdict),
    steps: queueStepSummary(row),
    meta:
      verdict === "OUT_OF_SCOPE" ? scopeReasonLabel(row.scope_reason) :
      verdict === "NEEDS_REVIEW" && review !== "PENDING"
        ? "Keputusan review dan hasil SOP berbeda; periksa detail"
        : "",
  };
};
