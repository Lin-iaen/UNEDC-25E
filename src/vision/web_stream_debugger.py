from __future__ import annotations

import numpy as np


class WebStreamDebugger:
    """Background HTTP MJPEG streamer for debug visualization.

    Starts a Flask daemon thread on ``0.0.0.0:5000``. The main loop calls
    ``update_frame(bgr_frame)`` every cycle, and the stream serves whatever
    the latest frame is.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        """Prepare the server but do not start it yet.

        Args:
            host: Bind address.
            port: HTTP port.
        """

    def update_frame(self, frame: np.ndarray) -> None:
        """Push the latest debug frame to the stream buffer.

        Args:
            frame: BGR image, shape (H, W, 3), dtype uint8.
        """

    def start(self) -> None:
        """Launch the Flask server in a daemon thread. Non-blocking."""

    def stop(self) -> None:
        """Shut down the HTTP server and join the thread."""
