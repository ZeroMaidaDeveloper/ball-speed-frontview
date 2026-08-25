// Ball Speed Viewer front-end.
//
// No build step, no external libraries — everything here talks to the
// two JSON endpoints exposed by viewer/app.py:
//   GET /api/videos             -> [{stem, filename, url, has_detections, ...}]
//   GET /api/detections/<stem>  -> detections.json contents, or 404
//
// and renders bbox/trajectory overlays on a <canvas> stacked exactly on
// top of the <video> element.

(() => {
  "use strict";

  const videoSelect = document.getElementById("video-select");
  const sourceSelect = document.getElementById("source-select");
  const labChannelSelect = document.getElementById("lab-channel-select");
  const videoMetaEl = document.getElementById("video-meta");
  const calibBadgeEl = document.getElementById("calib-badge");
  const stage = document.getElementById("stage");
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const labCanvas = document.getElementById("lab-canvas");
  const labCtx = labCanvas.getContext("2d");
  const noDataBanner = document.getElementById("no-data-banner");
  const btnPlay = document.getElementById("btn-play");
  const btnStepBack = document.getElementById("btn-step-back");
  const btnStepFwd = document.getElementById("btn-step-fwd");
  const seekBar = document.getElementById("seek-bar");
  const fpsLabel = document.getElementById("fps-label");
  const deliveriesPanel = document.getElementById("deliveries-panel");
  const deliveryList = document.getElementById("delivery-list");
  const deliveryEmptyNote = document.getElementById("delivery-empty-note");
  const rawModePanel = document.getElementById("raw-mode-panel");
  const rawModeNote = document.getElementById("raw-mode-note");
  const curvePanel = document.getElementById("curve-panel");
  const curveCanvas = document.getElementById("curve-canvas");
  const curveCtx = curveCanvas.getContext("2d");

  const SOURCE_STYLE = {
    yolo: { stroke: "#4da3ff", dashed: false, alpha: 1.0 },
    yolo_refined: { stroke: "#c77dff", dashed: false, alpha: 1.0 },
    classical: { stroke: "#4ddc8c", dashed: false, alpha: 1.0 },
    lab_scale: { stroke: "#ffb84d", dashed: false, alpha: 1.0 },
    auto_label: { stroke: "#ff4d94", dashed: false, alpha: 1.0 },
    zoom_track: { stroke: "#5df2d6", dashed: false, alpha: 1.0 },
    motion_rescue: { stroke: "#ffe14d", dashed: false, alpha: 1.0 },
    kalman_predicted: { stroke: "#ff6b6b", dashed: true, alpha: 0.55 },
  };
  const DEFAULT_FPS = 30.0;
  const TRAIL_SECONDS = 0.4; // how far back the trajectory trail reaches
  const MATCH_TOLERANCE_FACTOR = 0.75; // fraction of one frame period

  // --- state -----------------------------------------------------------

  let currentVideoMeta = null; // entry from /api/videos
  let sourceMode = "pipeline"; // "pipeline" | "motion_only" | "color_only" | "yolo"
  let detections = null; // parsed detections.json, or null if unavailable (pipeline mode)
  let flatFrames = []; // all frames across deliveries, sorted by t, tagged with deliveryId (pipeline mode)
  let rawCandidates = null; // parsed candidates_<mode>.json, or null (raw modes)
  let calibration = null; // /api/calibration/<stem> response, see pipeline/calibration.py
  let activeDeliveryId = null;
  let rafHandle = null;
  let labChannel = "none"; // "none" | "l" | "a" | "b" | "mask"
  let seeking = false; // true while the user is dragging #seek-bar
  // config.yaml: detection.lab_* -- overwritten from /api/lab_thresholds on
  // boot so the "mask" view matches pipeline/classical_detect.py exactly;
  // these are just sane fallbacks if that fetch hasn't landed yet.
  let labThresholds = { lab_l_channel_min: 40, lab_l_channel_max: 220, lab_a_channel_min: 150 };

  const RAW_MODE_LABELS = {
    motion_only: "MOG2 detections only (raw, untracked)",
    color_only: "LAB color detections only (raw, untracked)",
    lab_scale: "LAB-scale (CLAHE + hard color gate) only (raw, untracked)",
    yolo: "YOLO detections only (raw, untracked)",
    yolo_refined: "YOLO (heatmap-refined ROI zoom) only (raw, untracked)",
    auto_label: "Your auto-label pipeline (ROI + person-mask + choose())",
    zoom_track: "Zoom-track (fast-LAB-trigger + predicted-zoom tracking) -- feeds the fused pipeline",
  };

  // --- helpers -----------------------------------------------------------

  function isRawMode() {
    return sourceMode !== "pipeline";
  }

  // Whichever of detections/rawCandidates is active carries fps/width/height.
  function activeMeta() {
    return isRawMode() ? rawCandidates : detections;
  }

  function fps() {
    return (activeMeta() && activeMeta().fps) || DEFAULT_FPS;
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function buildFrameIndex(det) {
    const flat = [];
    if (!det || !Array.isArray(det.deliveries)) return flat;
    for (const delivery of det.deliveries) {
      for (const f of delivery.frames || []) {
        flat.push(Object.assign({ deliveryId: delivery.id }, f));
      }
    }
    flat.sort((a, b) => a.t - b.t);
    return flat;
  }

  // Nearest frame (any delivery) to time t, within tolerance seconds.
  function findNearestFrame(t, tolerance) {
    if (flatFrames.length === 0) return null;
    // Linear scan is plenty fast for the frame counts a single delivery
    // set will realistically have; binary search would be premature here.
    let best = null;
    let bestDelta = Infinity;
    for (const f of flatFrames) {
      const delta = Math.abs(f.t - t);
      if (delta < bestDelta) {
        bestDelta = delta;
        best = f;
      }
      if (f.t - t > tolerance && best && bestDelta <= tolerance) break;
    }
    return bestDelta <= tolerance ? best : null;
  }

  // All entries in `list` within `tolerance` seconds of `t` -- used for raw
  // candidate modes, where several detector candidates can legitimately
  // land in the same frame and all of them are worth showing at once.
  function findAllNear(list, t, tolerance) {
    return list.filter((f) => Math.abs(f.t - t) <= tolerance);
  }

  function framesForDelivery(deliveryId) {
    return flatFrames.filter((f) => f.deliveryId === deliveryId);
  }

  function findActiveDelivery(t) {
    if (!detections) return null;
    for (const d of detections.deliveries) {
      if (t >= d.start_t && t <= d.end_t) return d;
    }
    return null;
  }

  // --- canvas sizing -----------------------------------------------------

  function resizeOverlay() {
    const rect = stage.getBoundingClientRect();
    overlay.width = Math.max(1, Math.round(rect.width));
    overlay.height = Math.max(1, Math.round(rect.height));
    draw();
  }

  // --- drawing -------------------------------------------------------------

  function drawBox(entry, scaleX, scaleY) {
    const style = SOURCE_STYLE[entry.source] || SOURCE_STYLE.classical;
    const [x1, y1, x2, y2] = entry.bbox;
    ctx.save();
    ctx.globalAlpha = style.alpha;
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = 2;
    ctx.setLineDash(style.dashed ? [6, 4] : []);
    ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
    ctx.restore();

    // Small label with source + conf, for debugging/review.
    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = style.stroke;
    ctx.font = "11px -apple-system, sans-serif";
    const label = `${entry.source} ${(entry.conf ?? 0).toFixed(2)}`;
    ctx.fillText(label, x1 * scaleX, Math.max(10, y1 * scaleY - 4));
    ctx.restore();
  }

  // Draws one [top, bottom] point pair as a dashed segment + end markers
  // + label, in canvas pixel space. Shared by both calibration overlay
  // shapes below.
  function drawCalibSegment(points, scaleX, scaleY, color, label) {
    if (!Array.isArray(points) || points.length !== 2) return;
    const [px1, py1] = [points[0][0] * scaleX, points[0][1] * scaleY];
    const [px2, py2] = [points[1][0] * scaleX, points[1][1] * scaleY];

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(px1, py1);
    ctx.lineTo(px2, py2);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const [px, py] of [[px1, py1], [px2, py2]]) {
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    if (label) {
      ctx.font = "11px -apple-system, sans-serif";
      ctx.fillText(label, Math.min(px1, px2) + 8, Math.min(py1, py2) - 6);
    }
    ctx.restore();
  }

  // Draws the calibration reference points/lines (see
  // pipeline/calibration.py) directly on the overlay canvas -- these are
  // fixed, on-frame pixel coordinates from a single reference frame, not
  // tied to the current playback time, so this is drawn every frame
  // regardless of mode. `wickets_calib` always has explicit `points`; the
  // full pinhole `calib` type draws `near_points`/`far_points` when
  // present (added after the fact purely for this overlay -- see each
  // calib file's note) and falls back to badge-only (see
  // updateCalibBadge) when they're absent.
  function drawCalibrationOverlay(scaleX, scaleY) {
    if (!calibration) return;
    if (calibration.type === "wickets_calib") {
      const dist = typeof calibration.pixel_distance === "number" ? calibration.pixel_distance.toFixed(0) : "?";
      drawCalibSegment(calibration.points, scaleX, scaleY, "#00e5ff", `wicket calib: ${calibration.wicket_distance_m}m = ${dist}px`);
    } else if (calibration.type === "calib") {
      const nh = typeof calibration.near_stump_height_px === "number" ? calibration.near_stump_height_px.toFixed(0) : "?";
      const fh = typeof calibration.far_stump_height_px === "number" ? calibration.far_stump_height_px.toFixed(0) : "?";
      drawCalibSegment(calibration.near_points, scaleX, scaleY, "#4ddc8c", `near stump: ${nh}px`);
      drawCalibSegment(calibration.far_points, scaleX, scaleY, "#4ddc8c", `far stump: ${fh}px`);
    }
  }

  function draw() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const meta = activeMeta();
    if (!meta) return;

    const srcW = meta.width || video.videoWidth || overlay.width;
    const srcH = meta.height || video.videoHeight || overlay.height;
    if (!srcW || !srcH || !overlay.width || !overlay.height) return;

    const scaleX = overlay.width / srcW;
    const scaleY = overlay.height / srcH;

    drawCalibrationOverlay(scaleX, scaleY);

    const t = video.currentTime;
    const tol = (1 / fps()) * MATCH_TOLERANCE_FACTOR;

    if (isRawMode()) {
      // Raw candidate modes: no tracking/fusion has happened, so several
      // candidates can legitimately share a frame -- draw all of them,
      // no trail (there's no established trajectory to trail).
      if (!rawCandidates || !Array.isArray(rawCandidates.candidates)) return;
      for (const entry of findAllNear(rawCandidates.candidates, t, tol)) {
        drawBox(entry, scaleX, scaleY);
      }
      return;
    }

    if (!detections || flatFrames.length === 0) return;
    const current = findNearestFrame(t, tol);
    if (!current) return;

    // Trailing trajectory: previous frames of the same delivery, within
    // TRAIL_SECONDS of the current one, drawn as a fading polyline.
    const deliveryFrames = framesForDelivery(current.deliveryId);
    const trail = deliveryFrames.filter(
      (f) => f.t <= current.t && f.t >= current.t - TRAIL_SECONDS
    );

    if (trail.length > 1) {
      ctx.save();
      ctx.setLineDash([]);
      for (let i = 1; i < trail.length; i++) {
        const a = trail[i - 1];
        const b = trail[i];
        const age = (current.t - b.t) / TRAIL_SECONDS; // 0 = now, 1 = oldest
        ctx.globalAlpha = clamp(0.85 - age * 0.75, 0.08, 0.85);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(a.cx * scaleX, a.cy * scaleY);
        ctx.lineTo(b.cx * scaleX, b.cy * scaleY);
        ctx.stroke();
      }
      ctx.restore();
    }

    drawBox(current, scaleX, scaleY);
  }

  function tickWhilePlaying() {
    draw();
    renderLabFrame();
    updateActiveDelivery();
    if (!video.paused && !video.ended) {
      rafHandle = requestAnimationFrame(tickWhilePlaying);
    } else {
      rafHandle = null;
    }
  }

  // --- live LAB channel / color-mask view -----------------------------------
  //
  // Renders the ACTUAL LAB decomposition of the current video frame (not
  // just bbox overlays) so you can see what pipeline/classical_detect.py's
  // color signal looks like directly on the footage. Only active in the
  // "color_only" source mode. Converts sRGB -> CIE LAB -> OpenCV's 8-bit
  // LAB convention (L*255/100, a*+128, b*+128) to match config.yaml's
  // lab_* thresholds exactly.

  const LAB_WORK_WIDTH = 384; // processing resolution; canvas CSS scales it back up
  const _srgbToLinear = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    _srgbToLinear[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  const LAB_EPS = 0.008856451679035631;
  const LAB_KAPPA = 903.2962962962963;

  const _labOffscreen = document.createElement("canvas");
  const _labOffCtx = _labOffscreen.getContext("2d", { willReadFrequently: true });

  function isLabViewActive() {
    return (sourceMode === "color_only" || sourceMode === "lab_scale") && labChannel !== "none";
  }

  // Matches ball_label.py's render_lab(): JET colormap of the LAB
  // a-channel, standard MATLAB-style jet approximation (blue=low/cool,
  // red=high/warm). `v` is 0..255.
  function jetColor(v) {
    const x = clamp(v / 255, 0, 1);
    const r = clamp(1.5 - Math.abs(4 * x - 3), 0, 1);
    const g = clamp(1.5 - Math.abs(4 * x - 2), 0, 1);
    const b = clamp(1.5 - Math.abs(4 * x - 1), 0, 1);
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }

  function renderLabFrame() {
    const active = isLabViewActive();
    labCanvas.style.display = active ? "block" : "none";
    video.style.visibility = active ? "hidden" : "visible";
    if (!active || !video.videoWidth) return;

    const meta = activeMeta();
    const srcW = (meta && meta.width) || video.videoWidth;
    const srcH = (meta && meta.height) || video.videoHeight;
    const workW = LAB_WORK_WIDTH;
    const workH = Math.max(1, Math.round(workW * (srcH / srcW)));

    if (_labOffscreen.width !== workW || _labOffscreen.height !== workH) {
      _labOffscreen.width = workW;
      _labOffscreen.height = workH;
    }
    _labOffCtx.drawImage(video, 0, 0, workW, workH);

    let srcImageData;
    try {
      srcImageData = _labOffCtx.getImageData(0, 0, workW, workH);
    } catch (err) {
      return; // frame not decoded yet
    }
    const src = srcImageData.data;

    if (labCanvas.width !== workW || labCanvas.height !== workH) {
      labCanvas.width = workW;
      labCanvas.height = workH;
    }
    const outImageData = labCtx.createImageData(workW, workH);
    const out = outImageData.data;

    const lMin = labThresholds.lab_l_channel_min;
    const lMax = labThresholds.lab_l_channel_max;
    const aMin = labThresholds.lab_a_channel_min;

    for (let i = 0; i < src.length; i += 4) {
      const r = src[i];
      const g = src[i + 1];
      const b = src[i + 2];

      const rl = _srgbToLinear[r];
      const gl = _srgbToLinear[g];
      const bl = _srgbToLinear[b];

      const X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
      const Y = 0.2126729 * rl + 0.7151522 * gl + 0.072175 * bl;
      const Z = 0.0193339 * rl + 0.119192 * gl + 0.9503041 * bl;

      // D65 reference white: Xn=0.95047, Yn=1.0, Zn=1.08883.
      const xr = X / 0.95047;
      const yr = Y;
      const zr = Z / 1.08883;

      const fx = xr > LAB_EPS ? Math.cbrt(xr) : (LAB_KAPPA * xr + 16) / 116;
      const fy = yr > LAB_EPS ? Math.cbrt(yr) : (LAB_KAPPA * yr + 16) / 116;
      const fz = zr > LAB_EPS ? Math.cbrt(zr) : (LAB_KAPPA * zr + 16) / 116;

      const Lcv = clamp(Math.round((116 * fy - 16) * 2.55), 0, 255);
      const Acv = clamp(Math.round(500 * (fx - fy) + 128), 0, 255);
      const Bcv = clamp(Math.round(200 * (fy - fz) + 128), 0, 255);

      if (labChannel === "mask") {
        const isBallColor = Lcv >= lMin && Lcv <= lMax && Acv >= aMin;
        if (isBallColor) {
          out[i] = 255;
          out[i + 1] = 0;
          out[i + 2] = 255;
        } else {
          out[i] = r * 0.35;
          out[i + 1] = g * 0.35;
          out[i + 2] = b * 0.35;
        }
      } else if (labChannel === "jet") {
        // render_lab(): disp = clip((a-110)*3, 0, 255), then JET colormap.
        const disp = clamp(Math.round((Acv - 110) * 3), 0, 255);
        const [jr, jg, jb] = jetColor(disp);
        out[i] = jr;
        out[i + 1] = jg;
        out[i + 2] = jb;
      } else {
        const ov = labChannel === "l" ? Lcv : labChannel === "a" ? Acv : Bcv;
        out[i] = ov;
        out[i + 1] = ov;
        out[i + 2] = ov;
      }
      out[i + 3] = 255;
    }

    labCtx.putImageData(outImageData, 0, 0);
  }

  // --- delivery list / active highlighting --------------------------------

  function confidenceClass(conf) {
    if (conf === "high") return "confidence-high";
    if (conf === "medium") return "confidence-medium";
    return "confidence-low";
  }

  function renderDeliveryList() {
    deliveryList.innerHTML = "";
    const deliveries = (detections && detections.deliveries) || [];
    deliveryEmptyNote.style.display = deliveries.length === 0 ? "block" : "none";

    for (const d of deliveries) {
      const li = document.createElement("li");
      li.dataset.deliveryId = String(d.id);

      const row = document.createElement("div");
      row.className = "delivery-row";

      const speedEl = document.createElement("span");
      speedEl.className = "delivery-speed";
      const speed = typeof d.speed_kmh === "number" ? Math.round(d.speed_kmh) : "?";
      speedEl.textContent = `#${d.id}  ${speed} km/h`;

      const badge = document.createElement("span");
      badge.className = `confidence-badge ${confidenceClass(d.speed_confidence)}`;
      badge.textContent = d.speed_confidence || "unknown";

      row.appendChild(speedEl);
      row.appendChild(badge);
      li.appendChild(row);

      if (d.quality_flags && d.quality_flags.length) {
        const flags = document.createElement("div");
        flags.className = "quality-flags";
        flags.textContent = d.quality_flags.join(", ");
        li.appendChild(flags);
      }

      li.addEventListener("click", () => {
        video.currentTime = d.start_t;
        draw();
        updateActiveDelivery();
      });

      deliveryList.appendChild(li);
    }
  }

  function updateActiveDelivery() {
    const active = findActiveDelivery(video.currentTime);
    const newId = active ? active.id : null;
    if (newId === activeDeliveryId) return;
    activeDeliveryId = newId;

    for (const li of deliveryList.children) {
      li.classList.toggle("active", Number(li.dataset.deliveryId) === activeDeliveryId);
    }

    if (active && active.size_speed_curve && active.size_speed_curve.length > 0) {
      curvePanel.style.display = "block";
      drawCurve(active);
    } else {
      curvePanel.style.display = "none";
    }
  }

  // --- size_speed_curve mini chart -----------------------------------------

  function drawCurve(delivery) {
    const curve = delivery.size_speed_curve;
    // Match backing-store resolution to displayed size for crispness.
    const displayWidth = curveCanvas.clientWidth || 280;
    curveCanvas.width = displayWidth;
    const w = curveCanvas.width;
    const h = curveCanvas.height;

    curveCtx.clearRect(0, 0, w, h);

    const pad = { left: 40, right: 10, top: 10, bottom: 20 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const ts = curve.map((p) => p.t);
    const speeds = curve.map((p) => p.speed_kmh);
    const tMin = Math.min(...ts);
    const tMax = Math.max(...ts);
    let sMin = Math.min(...speeds);
    let sMax = Math.max(...speeds);
    if (sMin === sMax) {
      sMin -= 5;
      sMax += 5;
    } else {
      const margin = (sMax - sMin) * 0.15;
      sMin -= margin;
      sMax += margin;
    }

    const xAt = (t) => pad.left + (tMax === tMin ? 0.5 : (t - tMin) / (tMax - tMin)) * plotW;
    const yAt = (s) => pad.top + plotH - ((s - sMin) / (sMax - sMin)) * plotH;

    // Axes.
    curveCtx.strokeStyle = "#33383f";
    curveCtx.lineWidth = 1;
    curveCtx.beginPath();
    curveCtx.moveTo(pad.left, pad.top);
    curveCtx.lineTo(pad.left, pad.top + plotH);
    curveCtx.lineTo(pad.left + plotW, pad.top + plotH);
    curveCtx.stroke();

    curveCtx.fillStyle = "#9aa0aa";
    curveCtx.font = "10px -apple-system, sans-serif";
    curveCtx.fillText(sMax.toFixed(0), 4, pad.top + 4);
    curveCtx.fillText(sMin.toFixed(0), 4, pad.top + plotH);

    // Line.
    curveCtx.strokeStyle = "#4da3ff";
    curveCtx.lineWidth = 2;
    curveCtx.beginPath();
    curve.forEach((p, i) => {
      const x = xAt(p.t);
      const y = yAt(p.speed_kmh);
      if (i === 0) curveCtx.moveTo(x, y);
      else curveCtx.lineTo(x, y);
    });
    curveCtx.stroke();

    // Points.
    curveCtx.fillStyle = "#4da3ff";
    for (const p of curve) {
      curveCtx.beginPath();
      curveCtx.arc(xAt(p.t), yAt(p.speed_kmh), 2.5, 0, Math.PI * 2);
      curveCtx.fill();
    }
  }

  // --- video loading -------------------------------------------------------

  async function loadDetectionsFor(stem) {
    try {
      const res = await fetch(`/api/detections/${encodeURIComponent(stem)}`);
      if (!res.ok) {
        return null; // 404 (or other) -> treat as "no data yet"
      }
      return await res.json();
    } catch (err) {
      console.error("Failed to load detections:", err);
      return null;
    }
  }

  async function loadCalibrationFor(stem) {
    try {
      const res = await fetch(`/api/calibration/${encodeURIComponent(stem)}`);
      if (!res.ok) return { type: "none" };
      return await res.json();
    } catch (err) {
      console.error("Failed to load calibration:", err);
      return { type: "none" };
    }
  }

  function updateCalibBadge() {
    calibBadgeEl.classList.remove("calib-full", "calib-planar", "calib-none");
    if (!calibration || calibration.type === "none") {
      calibBadgeEl.textContent = "no calibration -- using flight_distance_m fallback";
      calibBadgeEl.classList.add("calib-none");
    } else if (calibration.type === "calib") {
      const f = typeof calibration.f_px === "number" ? calibration.f_px.toFixed(0) : "?";
      calibBadgeEl.textContent = `calibrated (pinhole, f_px=${f})`;
      calibBadgeEl.classList.add("calib-full");
    } else if (calibration.type === "wickets_calib") {
      const ppm = typeof calibration.pixels_per_meter === "number" ? calibration.pixels_per_meter.toFixed(1) : "?";
      calibBadgeEl.textContent = `wicket calib (planar, ${ppm} px/m)`;
      calibBadgeEl.classList.add("calib-planar");
    }
  }

  async function loadCandidatesFor(stem, mode) {
    try {
      const res = await fetch(`/api/candidates/${encodeURIComponent(stem)}/${encodeURIComponent(mode)}`);
      if (!res.ok) {
        return null; // 404 (or other) -> treat as "no data yet"
      }
      return await res.json();
    } catch (err) {
      console.error("Failed to load candidates:", err);
      return null;
    }
  }

  // (Re)loads whichever data source the "source-select" filter currently
  // points at, for the currently-selected video, and refreshes the UI
  // panels that depend on it. Called on video change AND on source-mode
  // change -- it does NOT touch the <video> element itself.
  async function loadSourceData() {
    activeDeliveryId = null;
    curvePanel.style.display = "none";

    if (!currentVideoMeta) return;
    const stem = currentVideoMeta.stem;

    const labViewAvailable = sourceMode === "color_only" || sourceMode === "lab_scale";
    labChannelSelect.style.display = labViewAvailable ? "inline-block" : "none";
    if (!labViewAvailable) {
      labChannel = "none";
      labChannelSelect.value = "none";
    }

    if (isRawMode()) {
      deliveriesPanel.style.display = "none";
      rawModePanel.style.display = "block";

      detections = null;
      flatFrames = [];
      rawCandidates = await loadCandidatesFor(stem, sourceMode);

      if (rawCandidates) {
        noDataBanner.style.display = "none";
        fpsLabel.textContent = `fps: ${rawCandidates.fps}`;
        rawModeNote.textContent = `${rawCandidates.candidates.length} raw candidates -- untracked, unfiltered by tracking/speed logic.`;
      } else {
        noDataBanner.textContent = `No ${RAW_MODE_LABELS[sourceMode] || sourceMode} data for this video yet.`;
        noDataBanner.style.display = "block";
        fpsLabel.textContent = `fps: ${DEFAULT_FPS} (fallback, no candidates file)`;
        rawModeNote.textContent = "";
      }
    } else {
      deliveriesPanel.style.display = "block";
      rawModePanel.style.display = "none";

      rawCandidates = null;
      detections = await loadDetectionsFor(stem);
      flatFrames = buildFrameIndex(detections);

      if (detections) {
        noDataBanner.style.display = "none";
        fpsLabel.textContent = `fps: ${detections.fps}`;
      } else {
        noDataBanner.textContent = "No tracking data for this video yet.";
        noDataBanner.style.display = "block";
        fpsLabel.textContent = `fps: ${DEFAULT_FPS} (fallback, no detections.json)`;
      }
    }

    renderDeliveryList();
    resizeOverlay();
    renderLabFrame();
  }

  async function selectVideo(meta) {
    currentVideoMeta = meta;
    updateSourceOptionAvailability(meta);

    videoMetaEl.textContent = `${(meta.size_bytes / (1024 * 1024)).toFixed(1)} MB${
      meta.duplicate_count ? ` (+${meta.duplicate_count} dup)` : ""
    }`;

    video.pause();
    video.src = meta.url;
    video.load();

    calibration = await loadCalibrationFor(meta.stem);
    updateCalibBadge();

    await loadSourceData();
  }

  async function loadVideoList() {
    const res = await fetch("/api/videos");
    const videos = await res.json();

    videoSelect.innerHTML = "";
    if (videos.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "No videos found";
      videoSelect.appendChild(opt);
      return;
    }

    for (const v of videos) {
      const opt = document.createElement("option");
      opt.value = v.stem;
      opt.textContent = `${v.filename}${v.has_detections ? "" : "  (no data yet)"}`;
      opt.dataset.meta = JSON.stringify(v);
      videoSelect.appendChild(opt);
    }

    videoSelect.addEventListener("change", () => {
      const opt = videoSelect.selectedOptions[0];
      if (!opt || !opt.dataset.meta) return;
      selectVideo(JSON.parse(opt.dataset.meta));
    });

    selectVideo(videos[0]);
  }

  // Annotate (not disable -- the data may just not be generated *yet*)
  // each source-select option with whether that mode's candidates file
  // actually exists for the current video, so it's obvious which filters
  // will show something vs. hit the "no data yet" banner.
  function updateSourceOptionAvailability(meta) {
    const available = new Set(["pipeline", ...(meta.available_candidate_modes || [])]);
    for (const opt of sourceSelect.options) {
      const has = available.has(opt.value);
      opt.textContent = `${RAW_MODE_LABELS[opt.value] || opt.textContent.replace(/ \(no data yet\)$/, "")}${
        has ? "" : "  (no data yet)"
      }`;
    }
  }

  sourceSelect.addEventListener("change", () => {
    sourceMode = sourceSelect.value;
    loadSourceData();
  });

  labChannelSelect.addEventListener("change", () => {
    labChannel = labChannelSelect.value;
    renderLabFrame();
  });

  async function loadLabThresholds() {
    try {
      const res = await fetch("/api/lab_thresholds");
      if (res.ok) labThresholds = await res.json();
    } catch (err) {
      console.error("Failed to load LAB thresholds, using fallback:", err);
    }
  }

  // --- transport controls ---------------------------------------------------

  btnPlay.addEventListener("click", () => {
    if (video.paused) video.play();
    else video.pause();
  });

  video.addEventListener("play", () => {
    btnPlay.textContent = "Pause";
    if (!rafHandle) rafHandle = requestAnimationFrame(tickWhilePlaying);
  });

  video.addEventListener("pause", () => {
    btnPlay.textContent = "Play";
    draw();
    renderLabFrame();
    updateActiveDelivery();
  });

  function stepFrames(delta) {
    video.pause();
    const step = delta / fps();
    const duration = isFinite(video.duration) ? video.duration : Infinity;
    video.currentTime = clamp(video.currentTime + step, 0, duration);
  }

  btnStepBack.addEventListener("click", () => stepFrames(-1));
  btnStepFwd.addEventListener("click", () => stepFrames(1));

  video.addEventListener("seeked", () => {
    draw();
    renderLabFrame();
    updateActiveDelivery();
  });
  video.addEventListener("timeupdate", () => {
    if (!seeking && isFinite(video.duration) && video.duration > 0) {
      seekBar.value = String(Math.round((video.currentTime / video.duration) * 1000));
    }
    if (video.paused) {
      draw();
      renderLabFrame();
      updateActiveDelivery();
    }
  });
  video.addEventListener("loadedmetadata", () => {
    resizeOverlay();
    renderLabFrame();
  });
  window.addEventListener("resize", resizeOverlay);

  // --- seek bar (replaces the native <video controls> scrub bar, which a
  // canvas painted on top of the video -- for the live LAB view -- would
  // visually cover anyway) ---------------------------------------------------

  seekBar.addEventListener("input", () => {
    seeking = true;
    if (isFinite(video.duration) && video.duration > 0) {
      video.currentTime = (Number(seekBar.value) / 1000) * video.duration;
    }
  });
  seekBar.addEventListener("change", () => {
    seeking = false;
  });

  // --- boot ------------------------------------------------------------------

  loadLabThresholds();
  loadVideoList();
})();
