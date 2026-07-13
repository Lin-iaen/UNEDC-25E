#!/usr/bin/env python3
"""
追踪误差调试脚本
================

基于矩形检测管线，当检测到目标后进入误差分析：
- 计算画面中心与矩形对角线中心的偏差 (Δx, Δy, d)
- d < TrackingThreshold 时视为追踪成功，误差归零
- 十字线 + 连接线直观显示偏差方向和距离
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.drivers import Camera
from src.vision import MjpegStreamer

# ══════════════════════════════════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════════════════════════════════

PARAMS = {
    "ExposureTime": 20000,
    "AnalogueGain": 1.0,
    "MinArea":        500,
    "MaxAspectRatio": 3.0,
    "MinContrast":    30,
    "GlobalThresh":   120,
    "TrackingThreshold": 20,   # 像素，低于此值视为追踪成功
}

CLR_OUTER   = (0, 255, 0)     # 绿 —— 外轮廓
CLR_INNER   = (255, 0, 0)     # 蓝 —— 内轮廓
CLR_DIAG    = (0, 0, 255)     # 红 —— 对角线
CLR_CENTER  = (0, 255, 255)   # 黄 —— 画面中心十字
CLR_RECT    = (255, 255, 0)   # 青 —— 矩形中心十字
CLR_LINE    = (255, 255, 255) # 白 —— 偏差连接线
CLR_REJECT  = (80, 80, 80)    # 灰 —— 被过滤轮廓
CLR_REJ_TXT = (200, 200, 200)
CLR_OK      = (0, 255, 0)     # 追踪成功
CLR_FAIL    = (0, 0, 255)

HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Tracking Test</title>
<style>
body{margin:0;font-family:monospace;background:#111;color:#ccc;display:flex;height:100vh}
#panel{width:320px;padding:10px;background:#1a1a1a;border-right:1px solid #333}
#main{flex:1;display:flex;align-items:center;justify-content:center;padding:4px}
img{max-width:100%;max-height:100%}
h3{margin:10px 0 6px;font-size:12px;border-bottom:1px solid #333;padding-bottom:3px}
.stats{font-size:10px;color:#666;margin-top:8px}
.stat{display:flex;justify-content:space-between;margin:2px 0;font-size:11px}
.stat .l{color:#aaa}.stat .v{color:#0f0;font-weight:bold}
label{display:block;margin:6px 0 2px;font-size:11px;color:#aaa}
input[type=range]{width:100%}
.btn{display:inline-block;margin:2px;padding:3px 8px;background:#333;color:#fff;
     border:1px solid #555;cursor:pointer;font-size:10px;text-decoration:none}
.btn:hover{background:#555}
</style></head><body>
<div id="panel">
<div class="stat"><span class="l">矩形数</span><span class="v" id="s_rects">--</span></div>
<div class="stat"><span class="l">FPS</span><span class="v" id="stat_fps">--</span></div>

<h3>📷 检测</h3>
<label>最小面积 <span class="val" id="v_MinArea">--</span></label>
<input type="range" id="sl_MinArea" min="100" max="10000" oninput="setParam('MinArea',parseInt(this.value))">
<label>最大长宽比 <span class="val" id="v_MaxAspectRatio">--</span></label>
<input type="range" id="sl_MaxAspectRatio" min="1" max="10" step="0.1" oninput="setParam('MaxAspectRatio',parseFloat(this.value))">
<label>最小对比度 <span class="val" id="v_MinContrast">--</span></label>
<input type="range" id="sl_MinContrast" min="10" max="100" oninput="setParam('MinContrast',parseInt(this.value))">

<h3>🧠 二值化</h3>
<label>阈值 <span class="val" id="v_GlobalThresh">--</span></label>
<input type="range" id="sl_GlobalThresh" min="0" max="255" oninput="setParam('GlobalThresh',parseInt(this.value))">

<h3>🎯 追踪</h3>
<label>追踪阈值 <span class="val" id="v_TrackingThreshold">--</span> px</label>
<input type="range" id="sl_TrackingThreshold" min="5" max="100" oninput="setParam('TrackingThreshold',parseInt(this.value))">


</div>
<div id="main">
<img src="/video_feed" id="stream">
</div>
<script>
const ALL_KEYS = ["ExposureTime","AnalogueGain","MinArea","MaxAspectRatio","MinContrast","GlobalThresh","TrackingThreshold"];
function toKey(k){return k.charAt(0).toLowerCase()+k.slice(1);}
function setParam(name,v){
  document.getElementById('v_'+name).textContent=v;
  fetch('/set?'+toKey(name)+'='+v);
}
function poll(){
  fetch('/stats').then(r=>r.json()).then(function(d){
    ALL_KEYS.forEach(function(k){
      var jk=toKey(k); var v=d[jk];
      if(v!==undefined && v!==null){
        document.getElementById('v_'+k).textContent=v;
        document.getElementById('sl_'+k).value=v;
      }
    });
    document.getElementById('stat_fps').textContent=d.fps;
    document.getElementById('s_rects').textContent=d.rects;
  });
}
poll();
setInterval(poll, 1000);
</script>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _is_rectangle(approx: np.ndarray) -> bool:
    return len(approx) == 4 and cv2.isContourConvex(approx)

def _aspect_ratio(cnt: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(cnt)
    if min(w, h) == 0: return 999.0
    return max(w / h, h / w)

def _contrast_check(gray, outer_approx, inner_approx):
    x, y, w, h = cv2.boundingRect(outer_approx)
    roi = gray[y:y+h, x:x+w]
    oa = outer_approx.copy(); ia = inner_approx.copy()
    oa[:,0,0] -= x; oa[:,0,1] -= y
    ia[:,0,0] -= x; ia[:,0,1] -= y
    mask_in = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask_in, [ia], -1, 255, cv2.FILLED)
    mask_out = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask_out, [oa], -1, 255, cv2.FILLED)
    tape = cv2.subtract(mask_out, mask_in)
    return cv2.mean(roi, mask=mask_in)[0] - cv2.mean(roi, mask=tape)[0]

def _order_corners(approx):
    pts = approx.reshape(4, 2)
    idx = np.lexsort((pts[:,0], pts[:,1]))
    pts = pts[idx]
    top = pts[:2][np.argsort(pts[:2,0])]
    bottom = pts[2:][np.argsort(pts[2:,0])[::-1]]
    return np.vstack([top, bottom])

def _center_of(ordered: np.ndarray) -> tuple[int, int]:
    """返回有序四边形的对角线交点（即中心）。"""
    cx = (ordered[0,0] + ordered[1,0] + ordered[2,0] + ordered[3,0]) // 4
    cy = (ordered[0,1] + ordered[1,1] + ordered[2,1] + ordered[3,1]) // 4
    return int(cx), int(cy)

def _draw_crosshair(img, cx, cy, size=10, color=(0,255,255), thickness=1):
    """在 (cx,cy) 处画十字线。"""
    h, w = img.shape[:2]
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)
    # 画小圆点标记中心
    cv2.circle(img, (cx, cy), 2, color, -1)


# ══════════════════════════════════════════════════════════════════════════════
# 主处理函数
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(frame: np.ndarray, params: dict) -> tuple[np.ndarray, int, dict]:
    """
    返回 (composed_image, rect_count, error_state)
    error_state = {
        "dx": int, "dy": int, "distance": float,
        "tracking": bool, "rect_center_x": int, "rect_center_y": int
    }
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    min_area   = int(params.get("MinArea", 500))
    max_ar     = float(params.get("MaxAspectRatio", 3.0))
    min_contrast = int(params.get("MinContrast", 30))
    global_thresh = int(params.get("GlobalThresh", 120))
    track_thresh  = int(params.get("TrackingThreshold", 20))

    # 二值化
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, global_thresh, 255, cv2.THRESH_BINARY_INV)

    debug_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    detection_view = frame.copy()
    # 在检测视图上画固定画面中心十字
    frame_cx, frame_cy = w // 2, h // 2
    _draw_crosshair(detection_view, frame_cx, frame_cy, 12, CLR_CENTER, 2)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug_view, contours, -1, (70, 70, 70), 1)

    error = {"dx": 0, "dy": 0, "distance": 0.0, "tracking": False,
             "rect_center_x": 0, "rect_center_y": 0}

    if hierarchy is None:
        return _compose_views(detection_view, debug_view, 0, error), 0, error

    hierarchy = hierarchy[0]
    found_pairs = []
    rejected = []

    for idx, (cnt, hinfo) in enumerate(zip(contours, hierarchy)):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if not _is_rectangle(approx):
            continue
        ar = _aspect_ratio(approx)
        if ar > max_ar:
            rejected.append((approx, f"AR={ar:.1f}>{max_ar}"))
            continue
        child_idx = hinfo[2]
        if child_idx == -1:
            continue
        child_cnt = contours[child_idx]
        child_epsilon = 0.02 * cv2.arcLength(child_cnt, True)
        child_approx = cv2.approxPolyDP(child_cnt, child_epsilon, True)
        if not _is_rectangle(child_approx):
            continue
        child_ar = _aspect_ratio(child_approx)
        if child_ar > max_ar:
            rejected.append((approx, f"AR={child_ar:.1f}>{max_ar}"))
            continue
        contrast = _contrast_check(gray, approx, child_approx)
        if contrast <= min_contrast:
            rejected.append((approx, f"C={contrast:.0f}<{min_contrast}"))
            continue
        found_pairs.append((approx, child_approx))

    # 绘制
    for cnt, reason in rejected:
        cv2.drawContours(detection_view, [cnt], -1, CLR_REJECT, 2)
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        cv2.putText(detection_view, reason, (rx + rw + 4, ry + rh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_REJ_TXT, 1, cv2.LINE_AA)

    for outer, inner in found_pairs:
        cv2.drawContours(detection_view, [outer], -1, CLR_OUTER, 2)
        cv2.drawContours(detection_view, [inner], -1, CLR_INNER, 2)
        ordered = _order_corners(inner)
        cv2.line(detection_view, tuple(ordered[0]), tuple(ordered[2]), CLR_DIAG, 2)
        cv2.line(detection_view, tuple(ordered[1]), tuple(ordered[3]), CLR_DIAG, 2)

        # 矩形中心
        rect_cx, rect_cy = _center_of(ordered)
        _draw_crosshair(detection_view, rect_cx, rect_cy, 10, CLR_RECT, 2)

        # 偏差连接线
        cv2.line(detection_view, (frame_cx, frame_cy), (rect_cx, rect_cy), CLR_LINE, 1)

        # 误差计算
        dx = rect_cx - frame_cx
        dy = rect_cy - frame_cy
        dist = np.sqrt(dx**2 + dy**2)
        tracking = dist <= track_thresh

        error = {
            "dx": dx, "dy": dy, "distance": round(dist, 1),
            "tracking": tracking,
            "rect_center_x": rect_cx, "rect_center_y": rect_cy,
        }

        # 在连接线上标注偏差值
        mid_x = (frame_cx + rect_cx) // 2
        mid_y = (frame_cy + rect_cy) // 2
        label = f"Δx:{dx:+d} Δy:{dy:+d}"
        if tracking:
            label = "TRACK ✓"
            cv2.putText(detection_view, label, (mid_x - 30, mid_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_OK, 2, cv2.LINE_AA)
        else:
            cv2.putText(detection_view, label, (mid_x - 30, mid_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_LINE, 1, cv2.LINE_AA)

    composed = _compose_views(detection_view, debug_view, len(found_pairs), error)
    return composed, len(found_pairs), error


def _compose_views(detection, debug, rect_count, error) -> np.ndarray:
    """并排合成，左上角显示追踪状态。"""
    combined = np.hstack([detection, debug])
    h, w = combined.shape[:2]

    if error["tracking"]:
        text = f"TRACK ✓  d:{error['distance']}"
        colour = CLR_OK
    elif rect_count > 0:
        text = f"Δx:{error['dx']:+d} Δy:{error['dy']:+d} d:{error['distance']}"
        colour = CLR_FAIL
    else:
        text = f"R:{rect_count}"
        colour = (0, 0, 255)

    cv2.putText(combined, text, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 路由工厂
# ══════════════════════════════════════════════════════════════════════════════

def make_route_stats():
    from flask import jsonify
    import datetime
    def handler(**kwargs):
        return jsonify({
            "fps": round(_fps, 1),
            "rects": _rect_count,
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "dx": _latest_error["dx"],
            "dy": _latest_error["dy"],
            "distance": _latest_error["distance"],
            "tracking": _latest_error["tracking"],
            **{k[0].lower() + k[1:]: v for k, v in PARAMS.items()},
        })
    return handler


def make_route_set(cam: Camera):
    from flask import jsonify, request
    def handler(**kwargs):
        for key in request.args:
            raw = request.args[key]
            try:
                val = float(raw)
                val = int(val) if val == int(val) and "." not in raw else val
            except ValueError:
                continue
            matched = None
            for pk in PARAMS:
                if pk.lower() == key.lower():
                    matched = pk; break
            if matched:
                PARAMS[matched] = val
                if matched in ("ExposureTime", "AnalogueGain"):
                    cam.set_params({matched: val})
        return jsonify({"ok": True})
    return handler


def make_route_reset_ae(cam: Camera):
    from flask import jsonify
    def handler(**kwargs):
        cam.set_params({"AeEnable": True})
        print("AE re-enabled")
        return jsonify({"ok": True})
    return handler


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

_frame_count = 0
_fps = 0.0
_last_ts = time.perf_counter()
_latest_error = {"dx": 0, "dy": 0, "distance": 0.0, "tracking": False,
                 "rect_center_x": 0, "rect_center_y": 0}
_rect_count = 0


def main() -> None:
    global _frame_count, _fps, _last_ts, _latest_error, _rect_count

    cam = Camera(vflip=True)
    cam.start()

    time.sleep(1.5)
    try:
        md = cam._cam.capture_metadata()
        PARAMS["ExposureTime"] = md.get("ExposureTime", 20000)
        PARAMS["AnalogueGain"] = md.get("AnalogueGain", 1.0)
        cam.set_params({"ExposureTime": PARAMS["ExposureTime"],
                        "AnalogueGain": PARAMS["AnalogueGain"]})
        print(f"AE locked: Expo={PARAMS['ExposureTime']}us, "
              f"Gain={PARAMS['AnalogueGain']:.2f}x")
    except Exception:
        print("Warning: could not read AE metadata")

    def frame_provider() -> np.ndarray | None:
        global _frame_count, _fps, _last_ts, _latest_error, _rect_count
        frame = cam.read()
        if frame is None:
            return None
        try:
            result, _rect_count, _latest_error = process_frame(frame, PARAMS)
        except Exception:
            import traceback; traceback.print_exc()
            result = np.hstack([frame, np.zeros_like(frame)])
            _rect_count = -1
        _frame_count += 1
        now = time.perf_counter()
        if now - _last_ts >= 1.0:
            _fps = _frame_count / (now - _last_ts)
            _frame_count = 0; _last_ts = now
        return result

    streamer = MjpegStreamer(
        frame_provider=frame_provider,
        port=5000,
        custom_template=HTML_PAGE,
        custom_routes={
            "/stats":     make_route_stats(),
            "/set":       make_route_set(cam),
            "/reset_ae":   make_route_reset_ae(cam),
        },
    )
    streamer.start()
    print("Tracking test ready at http://0.0.0.0:5000")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down ...")

    streamer.stop()
    cam.release()


if __name__ == "__main__":
    main()
