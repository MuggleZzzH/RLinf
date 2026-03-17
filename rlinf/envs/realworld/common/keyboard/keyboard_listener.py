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
import pathlib
import threading
import time
import warnings

# Default path for the file-based keyboard signal used in headless / Ray
# environments where pynput cannot capture keystrokes.
_DEFAULT_SIGNAL_FILE = "/tmp/rlinf_keyboard_signal"


class KeyboardListener:
    """Keyboard listener that **always** supports file-based input.

    The file-based backend is always active so that headless / Ray remote
    environments can always send keypresses via::

        echo c > /tmp/rlinf_keyboard_signal   # send 'c' (success)
        echo b > /tmp/rlinf_keyboard_signal   # send 'b' (failure)

    In addition, ``pynput`` is tried as a *bonus* backend when a working
    X11 / Wayland display is detected.  Both backends feed into the same
    internal queue, so either mechanism can be used at any time.

    The signal file path can be overridden via the ``RLINF_KEYBOARD_SIGNAL``
    environment variable.
    """

    def __init__(self):
        self.state_lock = threading.Lock()
        self.latest_data = {"key": None, "pending_key": None}
        self.last_intervene = 0

        # ---- File-based backend (always active) ----
        self._signal_file = pathlib.Path(
            os.environ.get("RLINF_KEYBOARD_SIGNAL", _DEFAULT_SIGNAL_FILE)
        )
        # Clear any stale content
        self._signal_file.write_text("")
        self._file_thread = threading.Thread(
            target=self._poll_signal_file, daemon=True
        )
        self._file_thread.start()

        # ---- pynput backend (best-effort) ----
        pynput_ok = self._try_pynput()
        if pynput_ok:
            warnings.warn(
                f"Keyboard: pynput active.  File-based fallback also available "
                f"via:  echo <key> > {self._signal_file}",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Keyboard: pynput unavailable (headless / no X11).  "
                f"Use file-based input:\n"
                f"  echo c > {self._signal_file}   (success)\n"
                f"  echo b > {self._signal_file}   (failure)",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # pynput backend
    # ------------------------------------------------------------------

    def _try_pynput(self) -> bool:
        """Attempt to start a pynput listener.  Return True on success."""
        # On Linux, pynput requires an X11/Wayland display to receive events.
        display = os.environ.get("DISPLAY", "")
        wayland = os.environ.get("WAYLAND_DISPLAY", "")
        if not display and not wayland:
            return False

        try:
            from pynput import keyboard

            listener = keyboard.Listener(
                on_press=self.on_key_press, on_release=self.on_key_release
            )
            listener.start()
            time.sleep(0.1)
            if not listener.is_alive():
                return False

            # Verify pynput can actually talk to the display server by sending
            # a synthetic keypress and checking if it arrives.
            try:
                controller = keyboard.Controller()
                with self.state_lock:
                    self.latest_data["pending_key"] = None
                controller.press(keyboard.Key.shift)
                controller.release(keyboard.Key.shift)
                time.sleep(0.15)
                with self.state_lock:
                    got = self.latest_data["pending_key"]
                    self.latest_data["pending_key"] = None  # clear probe
                if got is None:
                    # Listener thread is alive but not receiving events.
                    listener.stop()
                    return False
            except Exception:
                # Controller failed → display is not functional.
                listener.stop()
                return False

            self.listener = listener
            return True
        except Exception:
            return False

    def on_key_press(self, key):
        key_char = key.char if hasattr(key, "char") else str(key)
        with self.state_lock:
            self.latest_data["key"] = key_char
            self.latest_data["pending_key"] = key_char

    def on_key_release(self, key):
        with self.state_lock:
            self.latest_data["key"] = None

    # ------------------------------------------------------------------
    # File-based backend
    # ------------------------------------------------------------------

    def _poll_signal_file(self):
        """Background thread that polls the signal file for keypresses."""
        while True:
            try:
                if self._signal_file.exists():
                    content = self._signal_file.read_text().strip()
                    if content:
                        # Consume: clear the file immediately
                        self._signal_file.write_text("")
                        with self.state_lock:
                            self.latest_data["key"] = content
                            self.latest_data["pending_key"] = content
            except Exception:
                pass
            time.sleep(0.05)  # 50 ms poll interval

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_key(self) -> str | None:
        """Return and consume the latest keypress event."""
        with self.state_lock:
            pending_key = self.latest_data["pending_key"]
            self.latest_data["pending_key"] = None
            return pending_key
