#!/usr/bin/env python3
"""
IMX219 Camera Parameter Explorer + MJPEG Stream
===============================================

Lists all camera controls and sensor modes, then starts an HTTP server with a
live MJPEG preview.  Change parameters via the web UI or direct URL queries.

Usage:
    python tests/test_camera_params.py
    → open http://<pi-ip>:5000 in a browser, adjust sliders, observe results.

Query-string API (non-interactive):
    curl "http://<pi-ip>:5000/set?ExposureTime=30000&Brightness=0.3"

Preset scenarios:
    /set?preset=lowlight     high exposure + gain
    /set?preset=bright       low exposure + gain
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from picamera2 import Picamera2

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("camera_params")


# ── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__)
_camera: Picamera2 | None = None
_lock = threading.Lock()
_controls: dict = {}
_overlay_info: dict[str, Any] = {}
_vflip: bool = True
_hflip: bool = False
_latest_jpeg: bytes = b""
_manual_params: set = set()  # params the user has touched

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

PRESETS = {
    "lowlight": {"ExposureTime": 60000, "AnalogueGain": 8.0, "Brightness": 0.3},
    "bright":   {"ExposureTime": 5000, "AnalogueGain": 1.0, "Brightness": -0.2},
    "default":  {"ExposureTime": 20000, "AnalogueGain": 1.0, "Brightness": 0.0},
    "highcontrast": {"Contrast": 4.0, "Saturation": 2.0, "Sharpness": 4.0},
    "vivid":    {"Contrast": 2.0, "Saturation": 4.0, "Sharpness": 2.0},
}

HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>IMX219 Explorer</title>
<style>
body{margin:0;font-family:monospace;background:#111;color:#ccc;display:flex;height:100vh}
#panel{width:360px;overflow-y:auto;padding:12px;background:#1a1a1a;border-right:1px solid #333}
img{max-width:100%;height:auto;display:block}
#main{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:8px}
label{display:block;margin:8px 0 2px;font-size:12px;color:#aaa}
input[type=range]{width:100%}
input[type=text]{width:60px;background:#333;border:1px solid #555;color:#fff;padding:2px}
.slider-row{display:flex;align-items:center;gap:8px}
.val{font-size:11px;color:#0f0;min-width:50px}
.btn{display:inline-block;margin:2px;padding:4px 10px;background:#333;color:#fff;border:1px solid #555;cursor:pointer;font-size:11px;text-decoration:none}
.btn:hover{background:#555}
h3{margin:12px 0 6px;font-size:13px;border-bottom:1px solid #333;padding-bottom:4px}
.stats{font-size:10px;color:#666;margin-top:4px}
</style></head><body>
<div id="panel">
<h3>IMX219 Controls</h3>

<label>ExposureTime <span class="val" id="v_ExposureTime">{{vals.ExposureTime}}</span></label>
<div class="slider-row">
<input type="range" min="1" max="66666" value="{{vals.ExposureTime}}" oninput="setVal('ExposureTime',this.value)">
<a class="btn" href="/reset/ae" style="margin-left:4px">AUTO</a>
</div>

<label>AnalogueGain <span class="val" id="v_AnalogueGain">{{vals.AnalogueGain}}</span></label>
<div class="slider-row">
<input type="range" min="1" max="16" step="0.1" value="{{vals.AnalogueGain}}" oninput="setVal('AnalogueGain',this.value)">
<a class="btn" href="/reset/ae" style="margin-left:4px">AUTO</a>
</div>

<label>Brightness <span class="val" id="v_Brightness">{{vals.Brightness}}</span></label>
<div class="slider-row">
<input type="range" min="-1" max="1" step="0.05" value="{{vals.Brightness}}" oninput="setVal('Brightness',this.value)">
</div>

<label>Contrast <span class="val" id="v_Contrast">{{vals.Contrast}}</span></label>
<div class="slider-row">
<input type="range" min="0" max="32" step="0.1" value="{{vals.Contrast}}" oninput="setVal('Contrast',this.value)">
</div>

<label>Saturation <span class="val" id="v_Saturation">{{vals.Saturation}}</span></label>
<div class="slider-row">
<input type="range" min="0" max="32" step="0.1" value="{{vals.Saturation}}" oninput="setVal('Saturation',this.value)">
</div>

<label>Sharpness <span class="val" id="v_Sharpness">{{vals.Sharpness}}</span></label>
<div class="slider-row">
<input type="range" min="0" max="16" step="0.1" value="{{vals.Sharpness}}" oninput="setVal('Sharpness',this.value)">
</div>

<label>ExposureValue <span class="val" id="v_ExposureValue">{{vals.ExposureValue}}</span></label>
<div class="slider-row">
<input type="range" min="-8" max="8" step="0.5" value="{{vals.ExposureValue}}" oninput="setVal('ExposureValue',this.value)">
</div>

<h3>Presets</h3>
<a class="btn" href="/set?preset=lowlight">Low Light</a>
<a class="btn" href="/set?preset=bright">Bright</a>
<a class="btn" href="/set?preset=default">Default</a>
<a class="btn" href="/set?preset=highcontrast">H Contrast</a>
<a class="btn" href="/set?preset=vivid">Vivid</a>

<h3>Sensor Modes</h3>
<a class="btn" href="/mode/0">640×480 @200fps</a>
<a class="btn" href="/mode/1">1640×1232 @81fps</a>
<a class="btn" href="/mode/2">1920×1080 @47fps</a>
<a class="btn" href="/mode/3">3280×2464 @21fps</a>

<h3>Actions</h3>
<a class="btn" href="/capture">📷 Capture Photo</a>
<a class="btn" href="/flip/v">↕ V-Flip</a>
<a class="btn" href="/flip/h">↔ H-Flip</a>
<div class="stats">
Frame: <span id="fps">--</span> FPS | <span id="fcount">0</span> frames<br>
Mode: {{vals.mode}} | Size: {{vals.size}}<br>
Timestamp: <span id="ts">--</span>
</div>
</div>
<div id="main">
<img src="/video_feed" id="stream">
</div>
<script>
function setVal(name, v){
  document.getElementById('v_'+name).textContent=v;
  fetch('/set?'+name+'='+v);
}
document.getElementById('stream').onload=function(){
  fetch('/stats').then(r=>r.json()).then(d=>{
    document.getElementById('fps').textContent=d.fps;
    document.getElementById('fcount').textContent=d.frames;
    document.getElementById('ts').textContent=d.time;
  });
};
setInterval(function(){
  fetch('/stats').then(r=>r.json()).then(d=>{
    document.getElementById('fps').textContent=d.fps;
    document.getElementById('fcount').textContent=d.frames;
    document.getElementById('ts').textContent=d.time;
  });
},2000);
</script>
</body></html>"""


# ── Frame generator ──────────────────────────────────────────────────────────

_frame_count = 0
_fps = 0.0
_last_ts = time.perf_counter()


def generate_frames():
    global _frame_count, _fps, _latest_jpeg, _last_ts, _controls
    while True:
        with _lock:
            if _camera is None:
                time.sleep(0.1)
                continue
            frame = _camera.capture_array()
            metadata = _camera.capture_metadata()

            # Refresh auto-exposed runtime values (skip params user has manually set)
            if "ExposureTime" in metadata and "ExposureTime" not in _manual_params:
                _controls["ExposureTime"] = metadata["ExposureTime"]
            if "AnalogueGain" in metadata and "AnalogueGain" not in _manual_params:
                _controls["AnalogueGain"] = metadata["AnalogueGain"]

            if _vflip and _hflip:
                frame = cv2.flip(frame, -1)
            elif _vflip:
                frame = cv2.flip(frame, 0)
            elif _hflip:
                frame = cv2.flip(frame, 1)

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            h, w = bgr.shape[:2]
            exp_label = "auto" if "ExposureTime" not in _manual_params else "manual"
            gain_label = "auto" if "AnalogueGain" not in _manual_params else "manual"
            lines = [
                f"Exp:{_controls.get('ExposureTime','--')}us ({exp_label})",
                f"Gain:{_controls.get('AnalogueGain','--')}x ({gain_label})",
                f"Bri:{_controls.get('Brightness','--')}",
                f"{_camera.camera_properties.get('Model','')} {w}x{h}",
            ]
            y0 = 28
            for i, line in enumerate(lines):
                cv2.putText(bgr, line, (8, y0 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            _latest_jpeg = jpeg.tobytes()

        _frame_count += 1
        now = time.perf_counter()
        if now - _last_ts >= 1.0:
            _fps = _frame_count / (now - _last_ts)
            _frame_count = 0
            _last_ts = now

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + _latest_jpeg + b"\r\n")
        time.sleep(0.03)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    vals = {
        "ExposureTime": _controls.get("ExposureTime", ""),
        "AnalogueGain": _controls.get("AnalogueGain", ""),
        "Brightness": _controls.get("Brightness", ""),
        "Contrast": _controls.get("Contrast", ""),
        "Saturation": _controls.get("Saturation", ""),
        "Sharpness": _controls.get("Sharpness", ""),
        "ExposureValue": _controls.get("ExposureValue", ""),
        "mode": _overlay_info.get("mode", "--"),
        "size": _overlay_info.get("size", "--"),
    }
    return render_template_string(HTML_PAGE, vals=vals)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    return jsonify({
        "fps": round(_fps, 1),
        "frames": _frame_count,
        "time": datetime.now().strftime("%H:%M:%S"),
    })


@app.route("/set")
def set_param():
    """Adjust one or more controls: /set?ExposureTime=30000&Brightness=0.5
       Or use a preset: /set?preset=lowlight"""
    preset = request.args.get("preset")
    if preset and preset in PRESETS:
        params = PRESETS[preset]
        logger.info("Applying preset: %s", preset)
    else:
        params = {}
        for key in request.args:
            if key == "preset":
                continue
            try:
                val = request.args[key]
                if "." in val:
                    params[key] = float(val)
                else:
                    params[key] = int(val)
            except ValueError:
                params[key] = request.args[key]

    with _lock:
        if _camera is not None:
            try:
                _camera.set_controls(params)
                _controls.update(params)
                _manual_params.update(params.keys())
                logger.info("Controls updated: %s", params)
            except Exception as e:
                logger.warning("Control set failed: %s", e)
                return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "values": params})


@app.route("/mode/<int:mode_id>")
def set_mode(mode_id: int):
    global _camera
    with _lock:
        if _camera is None:
            return jsonify({"ok": False, "error": "no camera"}), 500
        try:
            available = _camera.sensor_modes
            if mode_id >= len(available):
                return jsonify({"ok": False, "error": f"mode {mode_id} not available"}), 400
            m = available[mode_id]
            _camera.stop()
            cfg = _camera.create_preview_configuration(
                sensor={"output_size": m["size"], "bit_depth": m["bit_depth"]},
            )
            _camera.configure(cfg)
            _camera.start()
            _overlay_info["mode"] = mode_id
            _overlay_info["size"] = str(m["size"])
            logger.info("Switched to sensor mode %d: %s @ %.1ffps", mode_id, m["size"], m["fps"])
        except Exception as e:
            logger.exception("Mode switch failed")
            return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "mode": mode_id, "size": list(m["size"]), "fps": m["fps"]})


@app.route("/flip/<axis>")
def toggle_flip(axis: str):
    global _vflip, _hflip
    if axis == "v":
        _vflip = not _vflip
    elif axis == "h":
        _hflip = not _hflip
    else:
        return jsonify({"ok": False, "error": "axis must be v or h"}), 400
    logger.info("Flip: vflip=%s hflip=%s", _vflip, _hflip)
    return jsonify({"ok": True, "hflip": _hflip, "vflip": _vflip})


@app.route("/reset/ae")
def reset_auto_exposure():
    global _manual_params
    with _lock:
        _manual_params.discard("ExposureTime")
        _manual_params.discard("AnalogueGain")
        if _camera is not None:
            _camera.set_controls({"AeEnable": True})
    logger.info("Auto exposure re-enabled")
    return jsonify({"ok": True, "msg": "Auto exposure restored"})


@app.route("/capture")
def capture():
    with _lock:
        if _camera is None:
            return jsonify({"ok": False, "error": "no camera"}), 500
        frame = _camera.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SAMPLES_DIR / f"paramtest_{ts}.jpg"
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info("Photo saved: %s", path)
    return jsonify({"ok": True, "path": str(path)})


# ── Entrypoint ───────────────────────────────────────────────────────────────


def print_capabilities(cam: Picamera2) -> None:
    print()
    print("=" * 60)
    print("  IMX219 Camera Capability Report")
    print("=" * 60)
    print(f"  Model:          {cam.camera_properties.get('Model', '?')}")
    print(f"  Resolution:     {cam.camera_properties.get('PixelArraySize', '?')}")
    print(f"  Pipeline:       {cam.camera_properties.get('PipelineHandler', '?')}")

    print()
    print("── Sensor Modes ──")
    print(f"  {'Mode':<5} {'Resolution':<14} {'Bit Depth':<10} {'Max FPS':<8}")
    print(f"  {'-'*5} {'-'*14} {'-'*10} {'-'*8}")
    for i, m in enumerate(cam.sensor_modes):
        print(f"  {i:<5} {str(m['size']):<14} {m['bit_depth']:<10} {m['fps']:<8.1f}")

    print()
    print("── Controls ──")
    print(f"  {'Control':<25} {'Min':<10} {'Max':<10} {'Default':<10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for k, v in sorted(cam.camera_controls.items()):
        min_v = str(v[0])
        max_v = str(v[1])
        def_v = str(v[2])
        print(f"  {k:<25} {min_v:<10} {max_v:<10} {def_v:<10}")
    print("=" * 60)


def main() -> None:
    global _camera, _controls, _overlay_info, _vflip, _hflip, _manual_params

    _vflip = True
    _hflip = False

    cam = Picamera2()
    print_capabilities(cam)

    cam.configure(cam.create_preview_configuration())
    cam.start()
    _camera = cam
    _overlay_info = {"mode": 0, "size": str(cam.sensor_modes[0]["size"])}

    # Wait for auto-exposure to settle, then read ACTUAL runtime values
    time.sleep(1.5)
    metadata = cam.capture_metadata()
    _controls = {
        "ExposureTime": metadata.get("ExposureTime", 20000),
        "AnalogueGain": metadata.get("AnalogueGain", 1.0),
        "Brightness": 0.0,
        "Contrast": 1.0,
        "Saturation": 1.0,
        "Sharpness": 1.0,
        "ExposureValue": 0.0,
    }
    logger.info("Actual runtime: ExposureTime=%d AnalogueGain=%.2f",
                _controls["ExposureTime"], _controls["AnalogueGain"])

    print()
    logger.info("IMX219 Explorer ready at http://0.0.0.0:5000")
    print()

    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down ...")
    finally:
        cam.stop()
        cam.close()
        logger.info("Done")


if __name__ == "__main__":
    main()
