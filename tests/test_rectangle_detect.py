#!/usr/bin/env python3
"""
矩形检测测试脚本 —— 双窗口调试版
===================================

检测由 1.8cm 黑色电工胶带在 A4 白纸上拼接的矩形，识别内外两层轮廓。
提供网页参数滑块、双窗口（检测/二值化）实时调试视图。

用法：
    python tests/test_rectangle_detect.py
    → 打开 http://<pi-ip>:5000，拖动滑块实时观察过滤效果
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
# 全局可调参数 —— 启动时用 AE 实测值填充曝光/增益，其余给初值
# ══════════════════════════════════════════════════════════════════════════════

PARAMS = {
    # 摄像机参数 —— start() 后用 capture_metadata() 覆盖实际值
    "ExposureTime": 20000,
    "AnalogueGain": 1.0,
    # 检测过滤参数
    "MinArea":        500,    # 最小轮廓面积 (px²)
    "MaxAspectRatio": 3.0,    # 最大长宽比（超过视为长条，丢弃）
    "MinWhiteRatio":  0.85,   # 内轮廓灰度均值比例（>此值才算白纸）
    "ThreshBlock":    31,     # 自适应阈值块大小（必须为奇数）
    "ThreshC":        3,      # 自适应阈值常数
}

# ══════════════════════════════════════════════════════════════════════════════
# 绘图颜色 (BGR)
# ══════════════════════════════════════════════════════════════════════════════

CLR_OUTER   = (0, 255, 0)     # 绿色 —— 外层轮廓（胶带外沿）
CLR_INNER   = (255, 0, 0)     # 蓝色 —— 内层轮廓（胶带内沿）
CLR_DIAG    = (0, 0, 255)     # 红色 —— 对角线
CLR_LABEL   = (255, 255, 0)   # 青色 —— 通过检测的标注
CLR_REJECT  = (80, 80, 80)    # 深灰 —— 被过滤掉的轮廓
CLR_TEXT    = (200, 200, 200) # 浅灰 —— 过滤原因文字
CLR_STATUS  = (0, 255, 255)   # 黄色 —— 状态栏

# ══════════════════════════════════════════════════════════════════════════════
# Web UI
# ══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Rectangle Detect</title>
<style>
body{margin:0;font-family:monospace;background:#111;color:#ccc;display:flex;height:100vh}
#panel{width:320px;overflow-y:auto;padding:10px;background:#1a1a1a;border-right:1px solid #333}
#main{flex:1;display:flex;align-items:center;justify-content:center;padding:4px}
img{max-width:100%;max-height:100%;object-fit:contain}
label{display:block;margin:6px 0 2px;font-size:11px;color:#aaa}
input[type=range]{width:100%}
.val{font-size:10px;color:#0f0;float:right}
.btn{display:inline-block;margin:2px;padding:3px 8px;background:#333;color:#fff;
     border:1px solid #555;cursor:pointer;font-size:10px;text-decoration:none}
.btn:hover{background:#555}
h3{margin:10px 0 6px;font-size:12px;border-bottom:1px solid #333;padding-bottom:3px}
.stats{font-size:10px;color:#666;margin-top:6px}
.section{margin-bottom:4px}
</style></head><body>
<div id="panel">

<h3>📷 Camera</h3>
<label>曝光时间 <span class="val" id="v_ExposureTime">--</span> µs</label>
<input type="range" id="sl_ExposureTime" min="39" max="66666"
       oninput="setParam('ExposureTime',parseInt(this.value))">

<label>模拟增益 <span class="val" id="v_AnalogueGain">--</span> ×</label>
<input type="range" id="sl_AnalogueGain" min="1" max="16" step="0.1"
       oninput="setParam('AnalogueGain',parseFloat(this.value))">
<a class="btn" href="/reset_ae">↻ 恢复自动曝光</a>

<h3>🔍 检测过滤</h3>
<label>最小面积 <span class="val" id="v_MinArea">--</span> px²</label>
<input type="range" id="sl_MinArea" min="100" max="10000"
       oninput="setParam('MinArea',parseInt(this.value))">

<label>最大长宽比 <span class="val" id="v_MaxAspectRatio">--</span></label>
<input type="range" id="sl_MaxAspectRatio" min="1" max="10" step="0.1"
       oninput="setParam('MaxAspectRatio',parseFloat(this.value))">

<label>最小白底比率 <span class="val" id="v_MinWhiteRatio">--</span></label>
<input type="range" id="sl_MinWhiteRatio" min="0.3" max="1.0" step="0.01"
       oninput="setParam('MinWhiteRatio',parseFloat(this.value))">

<h3>🧠 二值化</h3>
<label>阈值块大小 <span class="val" id="v_ThreshBlock">--</span></label>
<input type="range" id="sl_ThreshBlock" min="7" max="101" step="2"
       oninput="setParam('ThreshBlock',parseInt(this.value))">

<label>阈值常数 <span class="val" id="v_ThreshC">--</span></label>
<input type="range" id="sl_ThreshC" min="-10" max="20"
       oninput="setParam('ThreshC',parseInt(this.value))">

<div class="stats">
帧率: <span id="fps">--</span> FPS<br>
检测数: <span id="rects">--</span><br>
<span id="ts">--</span>
</div>
</div>

<div id="main">
<img src="/video_feed" id="stream">
</div>

<script>
const ALL_KEYS = ["ExposureTime","AnalogueGain","MinArea","MaxAspectRatio","MinWhiteRatio","ThreshBlock","ThreshC"];
// 将 key 的首字母转小写（Python dict key 的格式），因为 /stats 返回的 JSON key 是首字母小写
function toJsonKey(k){return k.charAt(0).toLowerCase()+k.slice(1);}

function setParam(name, val){
  document.getElementById('v_'+name).textContent = val;
  document.getElementById('sl_'+name).value = val;
  fetch('/set?'+toJsonKey(name)+'='+val);
}

function poll(){
  fetch('/stats').then(r=>r.json()).then(function(d){
    ALL_KEYS.forEach(function(k){
      var jk = toJsonKey(k);
      var v = d[jk];
      if(v !== undefined && v !== null){
        document.getElementById('v_'+k).textContent = v;
        document.getElementById('sl_'+k).value = v;
      }
    });
    document.getElementById('fps').textContent = d.fps;
    document.getElementById('ts').textContent = d.time;
    document.getElementById('rects').textContent = d.rects;
  });
}
poll();
setInterval(poll, 2000);
</script>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# 检测 pipeline
# ══════════════════════════════════════════════════════════════════════════════


def _is_rectangle(approx: np.ndarray) -> bool:
    """approxPolyDP 结果是否为凸四边形"""
    return len(approx) == 4 and cv2.isContourConvex(approx)


def _aspect_ratio(cnt: np.ndarray) -> float:
    """轮廓的宽高比，始终 ≥ 1（宽矮或瘦高都归一化到同一尺度）"""
    x, y, w, h = cv2.boundingRect(cnt)
    if min(w, h) == 0:
        return 999.0
    return max(w / h, h / w)


def _interior_white_ratio(gray: np.ndarray, contour: np.ndarray) -> float:
    """
    计算轮廓内部像素灰度均值与 255 的比值。
    直接用灰度图而非二值化结果——灰度图对光照变化更鲁棒。
    返回值 ∈ [0, 1]，越接近 1 表示内部越白。
    """
    # 创建掩膜：轮廓内部 = 255，外部 = 0
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 只用掩膜区域内的灰度值计算均值
    # 这是"看起来多此一举"的地方 —— 不用二值化而用原始灰度，
    # 因为自适应二值化对光照不均极其敏感，会把同一个白纸区域的一半判为黑，
    # 导致白色比率计算抖动。灰度均值天然对抗这种抖动。
    total = cv2.countNonZero(mask)
    if total == 0:
        return 0.0
    mean_val = cv2.mean(gray, mask=mask)[0]
    return mean_val / 255.0


def _order_corners(approx: np.ndarray) -> np.ndarray:
    """
    将 approxPolyDP 返回的四个顶点排序为：
    [左上, 右上, 右下, 左下]（顺时针）。
    approxPolyDP 不保证点顺序一致性，所以需要排序。
    """
    pts = approx.reshape(4, 2)
    # 先按 y 再按 x 排序 → 上半顶点在前、下半在后
    idx = np.lexsort((pts[:, 0], pts[:, 1]))
    pts = pts[idx]
    # 上半两个顶点按 x 排序 → 左上、右上
    top = pts[:2][np.argsort(pts[:2, 0])]
    # 下半两个顶点按 x 降序 → 右下、左下
    bottom = pts[2:][np.argsort(pts[2:, 0])[::-1]]
    return np.vstack([top, bottom])


def process_frame(frame: np.ndarray, params: dict) -> np.ndarray:
    """
    主检测函数：在一帧上做矩形检测 + 双窗口合成。
    返回合成后的 BGR 帧（检测结果 | 调试视图）。
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── 解读参数 ──────────────────────────────────────────────────────
    min_area   = int(params.get("MinArea", 500))
    max_ar     = float(params.get("MaxAspectRatio", 3.0))
    min_white  = float(params.get("MinWhiteRatio", 0.85))
    block_size = int(params.get("ThreshBlock", 31))
    thresh_c   = int(params.get("ThreshC", 3))

    # 确保 block_size 是奇数（OpenCV adaptiveThreshold 要求）
    if block_size % 2 == 0:
        block_size += 1

    # ── 二值化 ────────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,   # 胶带=黑→255，纸=白→0
        block_size, thresh_c,
    )

    debug_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # detection_view 从原始帧拷贝开始，不加任何预绘——只有检测结果才画
    detection_view = frame.copy()

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )

    # 调试视图：只画所有原始轮廓（灰色细线），不加任何文字
    cv2.drawContours(debug_view, contours, -1, (70, 70, 70), 1)

    if hierarchy is None:
        return _compose_views(detection_view, debug_view, 0)

    hierarchy = hierarchy[0]
    found_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    # 只记录"通过四边形检测但被后续过滤淘汰"的候选——减少无关噪点标注
    rejected: list[tuple[np.ndarray, str]] = []

    for idx, (cnt, hinfo) in enumerate(zip(contours, hierarchy)):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 不是四边形 → 太常见（墙上纹理、纸边角），不加标注
        if not _is_rectangle(approx):
            continue

        ar = _aspect_ratio(approx)
        # 长宽比过大 → 可能是纸边、桌沿，标注原因
        if ar > max_ar:
            rejected.append((approx, f"AR={ar:.1f}>{max_ar}"))
            continue

        # 找子轮廓（胶带内沿）
        child_idx = hinfo[2]
        if child_idx == -1:
            # 无子轮廓 → 可能是纸角而不是胶带框，不加标注
            continue

        child_cnt = contours[child_idx]
        child_epsilon = 0.02 * cv2.arcLength(child_cnt, True)
        child_approx = cv2.approxPolyDP(child_cnt, child_epsilon, True)

        if not _is_rectangle(child_approx):
            continue

        # 白色验证
        white_ratio = _interior_white_ratio(gray, child_approx)
        if white_ratio < min_white:
            rejected.append((approx, f"W={white_ratio:.2f}<{min_white}"))
            continue

        found_pairs.append((approx, child_approx))

    # ── 绘制检测视图：只画最终结果和相关过滤信息 ─────────────────────

    # 被过滤的候选（灰线 + 原因标注，放在轮廓外侧避免重叠）
    for cnt, reason in rejected:
        cv2.drawContours(detection_view, [cnt], -1, CLR_REJECT, 2)
        # 文字放在轮廓右下角外侧，避开与其他轮廓的文字碰撞
        x, y, cw, ch = cv2.boundingRect(cnt)
        cv2.putText(detection_view, reason, (x + cw + 4, y + ch // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_TEXT, 1, cv2.LINE_AA)

    # 通过的矩形
    for outer, inner in found_pairs:
        cv2.drawContours(detection_view, [outer], -1, CLR_OUTER, 2)
        cv2.drawContours(detection_view, [inner], -1, CLR_INNER, 2)
        ordered = _order_corners(inner)
        cv2.line(detection_view, tuple(ordered[0]), tuple(ordered[2]),
                 CLR_DIAG, 2)
        cv2.line(detection_view, tuple(ordered[1]), tuple(ordered[3]),
                 CLR_DIAG, 2)

    return _compose_views(detection_view, debug_view, len(found_pairs))


def _compose_views(
    detection: np.ndarray,
    debug: np.ndarray,
    rect_count: int,
) -> np.ndarray:
    """并排合成 + 左上角状态标记。"""
    combined = np.hstack([detection, debug])

    # 仅左上角一行状态
    colour = (0, 255, 0) if rect_count else (0, 0, 255)
    status = f"R:{rect_count}" if rect_count else "--"
    cv2.putText(combined, status, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

_frame_count = 0
_fps = 0.0
_last_ts = time.perf_counter()
_rect_count = 0


def main() -> None:
    global _frame_count, _fps, _last_ts, _rect_count

    cam = Camera(vflip=True)
    cam.start()

    # ── 读取自动曝光稳定后的实际值，然后锁定 ─────────────────────────
    # 不等 AE 稳定就直接用默认值的话，锁定的参数和实际不匹配，
    # 画面会突然跳变。所以先在后台采集几秒让 AE 收敛。
    time.sleep(2.0)
    # 通过 picamera2 底层的 capture_metadata 获取当前实际值
    try:
        from picamera2 import Picamera2
        metadata = cam._cam.capture_metadata()
        actual_exp = metadata.get("ExposureTime", 20000)
        actual_gain = metadata.get("AnalogueGain", 1.0)
        PARAMS["ExposureTime"] = actual_exp
        PARAMS["AnalogueGain"] = actual_gain
        # 锁定曝光/增益，防止 AE 后续改动导致二值化阈值漂移
        cam.set_params({
            "ExposureTime": actual_exp,
            "AnalogueGain": actual_gain,
        })
        print(f"AE locked: Exposure={actual_exp}us, Gain={actual_gain:.2f}x")
    except Exception:
        print("Warning: could not read AE metadata, using defaults")

    def frame_provider() -> np.ndarray | None:
        global _frame_count, _fps, _last_ts, _rect_count
        frame = cam.read()
        if frame is None:
            return None

        result = process_frame(frame, PARAMS)

        _frame_count += 1
        now = time.perf_counter()
        if now - _last_ts >= 1.0:
            _fps = _frame_count / (now - _last_ts)
            _frame_count = 0
            _last_ts = now

        return result

    streamer = MjpegStreamer(
        frame_provider=frame_provider,
        port=5000,
        custom_template=HTML_PAGE,
        custom_routes={
            "/stats": _make_route_stats(),
            "/set": _make_route_set(cam),
            "/reset_ae": _make_route_reset_ae(cam),
        },
    )
    streamer.start()
    print("Rectangle detector ready at http://0.0.0.0:5000")
    print("  Left panel  = parameter sliders")
    print("  Left half   = detection result + reject labels")
    print("  Right half  = binary threshold debug view")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down ...")

    streamer.stop()
    cam.release()


# ══════════════════════════════════════════════════════════════════════════════
# 路由工厂
# ══════════════════════════════════════════════════════════════════════════════

def _make_route_stats():
    """返回 /stats JSON：当前参数值 + 帧率"""
    def handler(**kwargs):
        import datetime
        return {
            "fps": round(_fps, 1),
            "rects": _rect_count,
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            **{k[0].lower() + k[1:]: v for k, v in PARAMS.items()},
        }
    return handler


def _make_route_set(cam: Camera):
    """处理 /set?key=value，更新 PARAMS，相机参数同步到硬件"""
    from flask import jsonify, request
    def handler(**kwargs):
        for key in request.args:
            raw = request.args[key]
            # 尝试转数字
            try:
                val = float(raw)
                val = int(val) if val == int(val) and "." not in raw else val
            except ValueError:
                continue
            # 找到 PARAMS 中匹配的 key（大小写不敏感）
            matched = None
            for pk in PARAMS:
                if pk.lower() == key.lower():
                    matched = pk
                    break
            if matched:
                PARAMS[matched] = val
                # 相机参数同步到硬件
                if matched in ("ExposureTime", "AnalogueGain"):
                    cam.set_params({matched: val})
        return jsonify({"ok": True})
    return handler


def _make_route_reset_ae(cam: Camera):
    """恢复自动曝光：解除 ExposureTime 和 AnalogueGain 手动锁定"""
    from flask import jsonify
    def handler(**kwargs):
        # 让 ISP 重新接管曝光和增益
        cam.set_params({"AeEnable": True})
        print("AE re-enabled")
        return jsonify({"ok": True})
    return handler


if __name__ == "__main__":
    main()
