import { apiFetchJson } from "../shared/api.js";

// site.js calls this explicitly; importing the controller does not touch the DOM.
export const initSetupPage = async () => {
  if (!document.body || !document.body.classList.contains("page-setup")) return;

  //populate all the setup by getting the elementid
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

  //refresh the values for the operations
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
