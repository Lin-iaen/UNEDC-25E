from abc import ABC, abstractmethod

import numpy as np


class BaseCamera(ABC):

    @abstractmethod
    def start(self) -> None:
        """Open the camera and begin streaming frames.
        Must be called before get_frame().
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the stream and release the camera resource."""

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Block until the next frame is available.

        Returns:
            np.ndarray: RGB image with shape (H, W, 3) and dtype uint8.
        """
