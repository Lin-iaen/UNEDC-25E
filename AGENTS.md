# AGENTS.md

## Architecture Principles

1. **Interface-Driven Development** — Every hardware device (camera, motor, sensor)
   must define an Abstract Base Class (ABC) in `abc.ABC` before any implementation.
   Consumers depend on the ABC, never on the concrete class.

2. **Environment Awareness** — OpenCV is headless-only. `cv2.imshow()`,
   `cv2.waitKey()`, `cv2.destroyAllWindows()` are forbidden. Debug visually via
   Flask MJPEG stream (`--stream`) or save-to-disk (`--capture`, `--test`).

3. **Decoupled Design** — Vision processing, control algorithm, and hardware driver
   modules must never import each other directly. The main program wires them
   together via dependency injection (constructor injection or config-based
   assembly). Each module knows only its own ABC dependencies.

## Environment

- **venv**: `./venv`, activate with `source venv/bin/activate`
- **Python**: 3.13
- **OpenCV**: headless — `cv2.imshow()` / `waitKey()` / `destroyAllWindows()` are unavailable
- **picamera2**: installed as Debian system package (not in venv). A `.pth` file at
  `venv/lib/python3.13/site-packages/system_dist.pth` points to
  `/usr/lib/python3/dist-packages` so the venv can import it.
- **Flask**: available in venv for MJPEG streaming.

## Camera

- Platform: Raspberry Pi 5 + PiSP camera stack
- Sensor: OV5647 (CSI interface)
- **Do NOT use `cv2.VideoCapture`** to access the CSI camera — the raw V4L2 device
  from `rp1-cfe` driver streams Bayer data that OpenCV cannot decode directly.
  Always use `picamera2.Picamera2`.

## Commands

```bash
source venv/bin/activate
python src/camera_demo.py --capture   # single still (full res)
python src/camera_demo.py --test 30   # burst + FPS report
python src/camera_demo.py --stream    # MJPEG HTTP on :5000
```

## Project Layout

| Path | Purpose |
|---|---|
| `src/` | Application code |
| `src/drivers/` | Hardware ABC interfaces (`BaseCamera`, `BaseCANMotor`) |
| `tests/` | Tests (empty) |
| `samples/` | Captured photos |
| `calibration_data/` | Camera calibration files |
| `docs/` | Documentation |
| `venv/` | Virtual environment |

## Gotchas

- This is a **Raspberry Pi 5** with the **PiSP** camera pipeline. Code that assumes
  legacy `raspistill` / `raspivid` or `bcm2835-v4l2` will not work.
- `picamera2` is not installable via `pip` on this system (PiWheels may time out).
  Use the `.pth` workaround; do not reinstall or modify `pyvenv.cfg`.
- Photos captured at full sensor resolution (2592×1944). The preview/test stream
  defaults to 640×480.
- No formatter / linter / typechecker is configured.
