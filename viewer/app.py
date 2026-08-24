"""Local review viewer for the ball-speed tracking pipeline.

Small Flask app (no build step, no external JS/CSS) that lets you:
  - pick a source video from Takneek/
  - scrub through it with a canvas overlay of the current bbox + trail,
    drawn from runs/<video_stem>/detections.json (per SCHEMA.md)
  - browse the per-video deliveries list and jump to a delivery
  - see the (optional) size_speed_curve for the active delivery

Run:
    python3 viewer/app.py
Then open http://127.0.0.1:5050/
"""

import json
import re
from pathlib import Path

import yaml
from flask import Flask, Response, abort, jsonify, render_template, send_from_directory

# --- paths -----------------------------------------------------------------

VIEWER_DIR = Path(__file__).resolve().parent
BASE_DIR = VIEWER_DIR.parent

with (BASE_DIR / "config.yaml").open() as _f:
    _CONFIG = yaml.safe_load(_f)

VIDEOS_DIR = BASE_DIR / _CONFIG["paths"]["videos_dir"]
RUNS_DIR = BASE_DIR / _CONFIG["paths"]["runs_dir"]

app = Flask(
    __name__,
    static_folder=str(VIEWER_DIR / "static"),
    template_folder=str(VIEWER_DIR / "templates"),
)

_DUP_SUFFIX_RE = re.compile(r"\(\d+\)\.mp4$", re.IGNORECASE)

# Raw per-detector candidate dumps the source filter in the viewer can
# switch to (see pipeline/run_diagnostics.py) -- each is a flat list of
# every candidate that detector produced, with NO tracking/fusion applied,
# for visually comparing what each signal alone finds.
_CANDIDATE_MODES = ("motion_only", "color_only", "lab_scale", "yolo", "yolo_refined", "auto_label", "zoom_track")


def _safe_stem(raw: str) -> str:
    """Collapse a user-supplied stem to a bare filename component,
    stripping any path separators to prevent path traversal."""
    return Path(raw).name


def list_videos():
    """Enumerate Takneek/*.mp4, de-duped by filesize (re-uploaded dupes
    carry a literal "(1)" suffix in the filename per SCHEMA.md), and
    flag whether runs/<stem>/detections.json exists yet."""
    if not VIDEOS_DIR.is_dir():
        return []

    by_size = {}
    for p in sorted(VIDEOS_DIR.glob("*.mp4")):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        by_size.setdefault(size, []).append(p)

    videos = []
    for size, paths in by_size.items():
        # Prefer the filename that does NOT carry a "(n)" dup suffix;
        # break remaining ties alphabetically for determinism.
        canonical = sorted(paths, key=lambda p: (bool(_DUP_SUFFIX_RE.search(p.name)), p.name))[0]
        stem = canonical.stem
        videos.append(
            {
                "stem": stem,
                "filename": canonical.name,
                "url": f"/media/{canonical.name}",
                "size_bytes": size,
                "duplicate_count": len(paths) - 1,
                "has_detections": (RUNS_DIR / stem / "detections.json").is_file(),
                "available_candidate_modes": [
                    m for m in _CANDIDATE_MODES if (RUNS_DIR / stem / f"candidates_{m}.json").is_file()
                ],
            }
        )

    videos.sort(key=lambda v: v["filename"])
    return videos


# --- routes ------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/videos")
def api_videos():
    return jsonify(list_videos())


@app.route("/api/lab_thresholds")
def api_lab_thresholds():
    """LAB range used by pipeline/classical_detect.py's color-based
    detection (config.yaml: detection.lab_*), so the viewer's live LAB
    channel/mask renderer stays in sync with what the backend detector
    actually thresholds on instead of a hardcoded copy in JS."""
    cfg_det = _CONFIG["detection"]
    return jsonify(
        {
            "lab_l_channel_min": cfg_det["lab_l_channel_min"],
            "lab_l_channel_max": cfg_det["lab_l_channel_max"],
            "lab_a_channel_min": cfg_det["lab_a_channel_min"],
        }
    )


@app.route("/api/detections/<stem>")
def api_detections(stem):
    """Return runs/<stem>/detections.json verbatim, or a 404 JSON body
    (never a crash) if the pipeline hasn't produced it yet."""
    safe_stem = _safe_stem(stem)
    path = RUNS_DIR / safe_stem / "detections.json"
    if not path.is_file():
        return (
            jsonify({"error": "no_detections", "stem": safe_stem, "message": "no detections.json for this video yet"}),
            404,
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"error": "bad_detections_file", "stem": safe_stem, "message": str(exc)}), 500
    return jsonify(data)


@app.route("/api/candidates/<stem>/<mode>")
def api_candidates(stem, mode):
    """Return runs/<stem>/candidates_<mode>.json verbatim (raw per-detector
    candidates, see pipeline/run_diagnostics.py), or a 404 JSON body if
    that mode hasn't been generated for this video yet."""
    safe_stem = _safe_stem(stem)
    if mode not in _CANDIDATE_MODES:
        return jsonify({"error": "unknown_mode", "mode": mode, "valid_modes": _CANDIDATE_MODES}), 400
    path = RUNS_DIR / safe_stem / f"candidates_{mode}.json"
    if not path.is_file():
        return (
            jsonify(
                {
                    "error": "no_candidates",
                    "stem": safe_stem,
                    "mode": mode,
                    "message": f"no candidates_{mode}.json for this video yet",
                }
            ),
            404,
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"error": "bad_candidates_file", "stem": safe_stem, "mode": mode, "message": str(exc)}), 500
    return jsonify(data)


@app.route("/media/<filename>")
def media(filename):
    """Serve a raw source video with HTTP range-request support so the
    <video> element can scrub. Flask's send_from_directory passes
    conditional=True through to send_file, which (Flask >= 2.0 /
    Werkzeug's file wrapper) handles Range/If-Range and returns 206
    Partial Content responses — verified manually with curl, see
    viewer notes."""
    safe_name = _safe_stem(filename)
    full_path = VIDEOS_DIR / safe_name
    if not full_path.is_file():
        abort(404)
    resp: Response = send_from_directory(VIDEOS_DIR, safe_name, conditional=True, mimetype="video/mp4")
    return resp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
