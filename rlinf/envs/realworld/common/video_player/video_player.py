# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import queue
import threading
import warnings

import cv2
import numpy as np


class VideoPlayer:
    def __init__(self, enable: bool = True):
        self.queue = queue.Queue()
        self.is_running = False
        if not enable:
            return
        self._run_thread = threading.Thread(target=self._play, daemon=True)
        self._run_thread.start()

    def put_frame(self, frame):
        if self.is_running:
            self.queue.put(frame)

    def _play(self):
        # Check for a usable display (DISPLAY or WAYLAND_DISPLAY must be set and non-empty).
        display = os.environ.get("DISPLAY", "")
        wayland = os.environ.get("WAYLAND_DISPLAY", "")
        if not display and not wayland:
            warnings.warn(
                "No display found (DISPLAY / WAYLAND_DISPLAY unset). "
                "VideoPlayer will not run. Set DISPLAY to enable."
            )
            return

        # Even with DISPLAY set, OpenCV may lack GUI support (headless build).
        try:
            # A lightweight probe: create and destroy a tiny named window.
            cv2.namedWindow("_probe", cv2.WINDOW_AUTOSIZE)
            cv2.destroyWindow("_probe")
        except cv2.error:
            warnings.warn(
                "OpenCV GUI backend unavailable (headless build?). "
                "VideoPlayer will not run. Rebuild OpenCV with GTK/Qt support to enable."
            )
            return

        self.is_running = True
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [v for k, v in img_array.items() if "full" not in k], axis=0
            )

            cv2.imshow("Cameras", frame)
            cv2.waitKey(1)
