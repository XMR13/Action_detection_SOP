const eventLabels = {
  roll_entered: "Roll masuk",
  cleaned_done: "Pembersihan selesai",
  labeled_done: "Label terpasang",
  roll_left: "Roll keluar",
  recording_ended: "Perekaman selesai",
};

export const evidenceEventLabel = (event) =>
  eventLabels[event.name] || String(event.label || event.name || "Bukti sesi").replaceAll("_", " ");

export const evidenceCaptionAt = (events, time) => events
  .filter((event) => Number.isFinite(event.playback_time_s)
    && time >= event.playback_time_s && time < event.playback_time_s + 2)
  .map(evidenceEventLabel).join(" · ");

export const mountSessionEvidence = ({ payload, startS, player, evidenceStrip, evidenceState, evidencePill }) => {
  const clips = (Array.isArray(payload.clips) ? payload.clips : []).filter((clip) => clip && clip.url);
  const combined = payload.combined_video;
  let useCombined = Boolean(combined && combined.playback_url);
  const events = (combined && Array.isArray(combined.events) ? combined.events : [])
    .filter((event) => event && Number.isFinite(event.playback_time_s));
  const annotatedUrl = payload.annotated_playback_url || payload.annotated_url;
  const finalStatus = payload.final_sop || (payload.sop && payload.sop.final && payload.sop.final.overall);
  const preferAnnotated = Boolean(annotatedUrl && finalStatus === "UNKNOWN");

  const setState = (text, className = "brand") => {
    if (evidenceState) {
      evidenceState.textContent = text;
      evidenceState.className = `pill ${className}`;
    }
    if (evidencePill) evidencePill.textContent = text;
  };
  const activate = (key) => {
    if (!evidenceStrip) return;
    evidenceStrip.querySelectorAll("button[data-clip-key]").forEach((button) => {
      button.classList.toggle("active", button.dataset.clipKey === key);
    });
  };
  const thumbnail = () => {
    player.replaceChildren();
    if (payload.thumbnail_url) {
      const image = document.createElement("img");
      image.src = payload.thumbnail_url;
      image.alt = "Bukti sesi";
      image.style.cssText = "width:100%;height:100%;object-fit:contain;border-radius:inherit";
      player.appendChild(image);
    } else {
      const message = document.createElement("span");
      message.className = "pill";
      message.textContent = "Tidak ada bukti video";
      player.appendChild(message);
    }
  };
  const showVideo = (url, mode, seekTo = 0) => {
    const current = player.querySelector("video");
    if (current && current.dataset.evidenceMode === mode && current.getAttribute("src") === url) {
      if (current.readyState >= 1) current.currentTime = seekTo;
      else current.addEventListener("loadedmetadata", () => { current.currentTime = seekTo; }, { once: true });
      return;
    }
    player.replaceChildren();
    const video = document.createElement("video");
    video.id = "detail-video";
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    video.dataset.evidenceMode = mode;
    video.style.cssText = "width:100%;height:100%;object-fit:contain;border-radius:inherit";
    const caption = document.createElement("span");
    caption.className = "detail-video-caption";
    caption.setAttribute("role", "status");
    caption.textContent = mode === "combined" ? "Memuat video bukti…" : "Memuat video…";
    player.append(video, caption);
    video.addEventListener("loadedmetadata", () => {
      if (!player.contains(video)) return;
      video.currentTime = Math.min(seekTo, Math.max(0, video.duration - 0.01));
      if (mode === "combined") setState("Video bukti sesi");
      updateCaption();
    }, { once: true });
    const updateCaption = () => {
      const text = mode === "combined" ? evidenceCaptionAt(events, video.currentTime) : "";
      if (caption.textContent !== text) caption.textContent = text;
      caption.hidden = !text;
    };
    video.addEventListener("timeupdate", updateCaption);
    video.addEventListener("seeked", updateCaption);
    video.addEventListener("error", () => {
      if (!player.contains(video)) return;
      if (mode === "combined") {
        useCombined = false;
        renderStrip();
        setState("Video sesi belum tersedia; bukti asli ditampilkan", "dir-b");
        if (clips.length) {
          showVideo(clips[0].playback_url || clips[0].url, "clip-0");
          activate("clip-0");
          return;
        }
      }
      thumbnail();
      setState("Video tidak dapat diputar; periksa bukti gambar", "dir-b");
    }, { once: true });
    video.src = url;
  };
  const addButton = (label, key, action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pill detail-clip-pill";
    button.dataset.clipKey = key;
    button.textContent = label;
    button.addEventListener("click", () => { action(); activate(key); });
    evidenceStrip.appendChild(button);
  };
  const renderStrip = () => {
    if (!evidenceStrip) return;
    evidenceStrip.replaceChildren();
    if (useCombined) {
      addButton("Video sesi", "combined", () => showVideo(combined.playback_url, "combined"));
      events.forEach((event, index) => {
        const sourceOffset = Number(event.source_time_s) - startS;
        const tag = Number.isFinite(sourceOffset) ? ` +${Math.max(0, sourceOffset).toFixed(1)}s` : "";
        addButton(evidenceEventLabel(event) + tag, `event-${index}`,
          () => showVideo(combined.playback_url, "combined", event.playback_time_s));
      });
    } else {
      clips.forEach((clip, index) => {
        const sourceOffset = Number(clip.event_time_s) - startS;
        const tag = Number.isFinite(sourceOffset) ? ` +${Math.max(0, sourceOffset).toFixed(1)}s` : "";
        addButton(evidenceEventLabel(clip) + tag, `clip-${index}`,
          () => showVideo(clip.playback_url || clip.url, `clip-${index}`));
      });
    }
    if (annotatedUrl) {
      addButton("Video sesi penuh", "annotated", () => {
        showVideo(annotatedUrl, "annotated");
        setState("Video sesi penuh");
      });
    }
  };
  renderStrip();
  if (preferAnnotated) {
    showVideo(annotatedUrl, "annotated");
    activate("annotated");
    setState("Video sesi penuh");
  } else if (useCombined) {
    showVideo(combined.playback_url, "combined");
    activate("combined");
    setState("Memuat video bukti…");
  } else if (clips.length) {
    showVideo(clips[0].playback_url || clips[0].url, "clip-0");
    activate("clip-0");
    setState(`${clips.length} klip bukti asli`);
  } else if (annotatedUrl) {
    showVideo(annotatedUrl, "annotated");
    activate("annotated");
    setState("Video sesi penuh");
  } else {
    thumbnail();
    setState(payload.thumbnail_url ? "Bukti gambar" : "Tidak ada bukti", "dir-b");
  }
};
