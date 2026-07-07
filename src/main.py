#!/usr/bin/env python3
"""Entrypoint for the electronics competition main loop.

State-machine driven pipeline:

  INIT → IDLE → VISION_SEARCH → CLOSED_LOOP_TRACKING → IDLE …
                   ↓                                      ↓
                 ERROR ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
"""

import logging
import signal
import sys
import time
from enum import Enum, auto

from src.drivers import BaseCamera, BaseCANMotor
from src.vision import BaseTracker, WebStreamDebugger

logger = logging.getLogger("main")


class State(Enum):
    INIT = auto()
    IDLE = auto()
    VISION_SEARCH = auto()
    CLOSED_LOOP_TRACKING = auto()
    ERROR = auto()


class ShutdownFlag:
    """Thread-safe flag shared across the module."""

    def __init__(self) -> None:
        self._value = False

    def set(self) -> None:
        self._value = True

    @property
    def is_set(self) -> bool:
        return self._value


def _handle_signal(signum: int, _frame, shutdown: ShutdownFlag) -> None:
    logger.warning("Signal %d received, shutting down ...", signum)
    shutdown.set()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_signal_handlers(shutdown: ShutdownFlag) -> None:
    signal.signal(signal.SIGINT, lambda s, f: _handle_signal(s, f, shutdown))
    signal.signal(signal.SIGTERM, lambda s, f: _handle_signal(s, f, shutdown))


def handle_tracking(state: State, result: dict) -> State:
    """Process tracking result and return the next state.

    Override this with real control logic later.
    """
    return state


def main() -> None:
    setup_logging()
    shutdown = ShutdownFlag()
    setup_signal_handlers(shutdown)

    state = State.INIT
    logger.info("System starting (state=%s)", state.name)

    # ── Hardware & module instances (stubs, wired via dependency injection) ──
    camera: BaseCamera | None = None
    motor: BaseCANMotor | None = None
    tracker: BaseTracker | None = None
    debugger = WebStreamDebugger()

    try:
        debugger.start()
        logger.info("WebStreamDebugger started")

        state = State.IDLE
        logger.info("Entering main loop (state=%s)", state.name)

        while not shutdown.is_set:
            # ── 1. Capture ──
            if camera is None:
                time.sleep(0.1)
                continue

            try:
                frame = camera.get_frame()
            except Exception:
                logger.exception("Camera get_frame failed")
                state = State.ERROR
                continue

            # ── 2. Vision ──
            result: dict = {}
            annotated: frame = frame  # fallback: raw frame if no tracker

            if tracker is not None:
                try:
                    result, annotated = tracker.process_frame(frame)
                except Exception:
                    logger.exception("Tracker process_frame failed")
                    state = State.ERROR
                    continue

            # ── 3. Push debug frame to web stream ──
            debugger.update_frame(annotated)

            # ── 4. State machine ──
            try:
                if state == State.IDLE:
                    if tracker is not None:
                        state = State.VISION_SEARCH
                        logger.info("Transition IDLE → VISION_SEARCH")

                elif state == State.VISION_SEARCH:
                    if result:
                        state = State.CLOSED_LOOP_TRACKING
                        logger.info("Transition VISION_SEARCH → CLOSED_LOOP_TRACKING")

                elif state == State.CLOSED_LOOP_TRACKING:
                    if not result:
                        state = State.IDLE
                        logger.info("Transition CLOSED_LOOP_TRACKING → IDLE (target lost)")
                    else:
                        state = handle_tracking(state, result)

                elif state == State.ERROR:
                    logger.warning("In ERROR state – waiting for manual recovery")
                    state = State.IDLE

            except Exception:
                logger.exception("State machine error")
                state = State.ERROR

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception:
        logger.exception("Unhandled exception in main loop")
    finally:
        # ── Graceful shutdown ──
        logger.info("Shutting down ...")

        if camera is not None:
            try:
                camera.stop()
                logger.info("Camera stopped")
            except Exception:
                logger.exception("Camera stop failed")

        if motor is not None:
            try:
                logger.info("Motor disconnected")
            except Exception:
                logger.exception("Motor disconnect failed")

        try:
            debugger.stop()
            logger.info("Debugger stopped")
        except Exception:
            logger.exception("Debugger stop failed")

        logger.info("System shutdown complete")


if __name__ == "__main__":
    main()
