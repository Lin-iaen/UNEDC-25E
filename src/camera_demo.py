#!/usr/bin/env python3
"""CSI camera demo for Raspberry Pi (headless OpenCV compatible).

Usage:
    python src/camera_demo.py --capture    # Single photo
    python src/camera_demo.py --stream     # MJPEG HTTP stream at http://<ip>:5000
    python src/camera_demo.py --test 30    # Capture 30 frames, show FPS
"""

import argparse
import io
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def capture_single(cam):
    cam.stop()
    cfg = cam.create_still_configuration()
    cam.configure(cfg)
    cam.start()
    frame = cam.capture_array()
    cam.stop()
    save_image(frame)


def save_image(frame):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SAMPLES_DIR / f"capture_{ts}.jpg"
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {path}")


def run_test(cam, count):
    print(f"Capturing {count} frames ...")
    frames = []
    start = time.perf_counter()
    for i in range(count):
        frame = cam.capture_array()
        frames.append(frame)
        print(f"  {i+1}/{count}  shape={frame.shape} dtype={frame.dtype}")
    elapsed = time.perf_counter() - start
    fps = count / elapsed
    print(f"\nResult: {count} frames in {elapsed:.2f}s = {fps:.1f} FPS")
    print(f"Frame shape: {frames[0].shape}")
    save_image(frames[-1])


def run_stream(cam, host="0.0.0.0", port=5000):
    from flask import Flask, Response

    app = Flask(__name__)

    def generate():
        while True:
            frame = cam.capture_array()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
            time.sleep(0.03)

    @app.route("/")
    def index():
        return '<img src="/video_feed" width="100%">'

    @app.route("/video_feed")
    def video_feed():
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"Stream ready at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)


def main():
    parser = argparse.ArgumentParser(description="CSI Camera Demo")
    parser.add_argument("--capture", action="store_true", help="Capture single photo")
    parser.add_argument("--stream", action="store_true", help="Start MJPEG HTTP stream")
    parser.add_argument("--test", type=int, nargs="?", const=30, metavar="N",
                        help="Capture N frames and report FPS (default 30)")
    args = parser.parse_args()

    if not any([args.capture, args.stream, args.test]):
        parser.print_help()
        return

    cam = Picamera2()
    try:
        cam.start()
        if args.capture:
            capture_single(cam)
        elif args.stream:
            run_stream(cam)
        elif args.test:
            run_test(cam, args.test)
    finally:
        cam.close()


if __name__ == "__main__":
    main()
