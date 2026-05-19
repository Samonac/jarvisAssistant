"""Wake Word Detection for Jarvis Assistant.

Listens continuously for "Hey Jarvis" using OpenWakeWord (lightweight model).
When detected, sends a notification to the frontend to activate the microphone.

Requires:
- USB microphone connected to the Raspberry Pi
- pip install openwakeword pyaudio numpy

This runs as a background thread within the Flask app, or can be run standalone.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

WAKE_WORDS = ["hey_jarvis", "hey jarvis", "jarvis"]
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms at 16kHz


class WakeWordDetector:
    """Listens for wake words using OpenWakeWord.

    When a wake word is detected, triggers a callback or adds to a notification queue.

    Attributes:
        enabled: Whether wake word detection is active.
        threshold: Detection confidence threshold (0-1).
        on_wake: Callback function called when wake word is detected.
    """

    def __init__(self, threshold: float = 0.5, on_wake=None):
        self.enabled = False
        self.threshold = threshold
        self.on_wake = on_wake
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._model = None
        self._available = False

        # Check if dependencies are available
        try:
            import pyaudio
            import numpy as np
            self._available = True
        except ImportError:
            logger.warning(
                "Wake word detection unavailable. Install: pip install openwakeword pyaudio numpy"
            )

    def is_available(self) -> bool:
        """Check if wake word detection dependencies are installed."""
        return self._available

    def start(self) -> bool:
        """Start listening for wake words in a background thread.

        Returns True if started successfully, False if dependencies missing.
        """
        if not self._available:
            return False

        if self._thread and self._thread.is_alive():
            return True

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        self.enabled = True
        logger.info("Wake word detection started (threshold: %.2f)", self.threshold)
        return True

    def stop(self) -> None:
        """Stop listening for wake words."""
        self._stop_event.set()
        self.enabled = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Wake word detection stopped")

    def _listen_loop(self) -> None:
        """Main listening loop — runs in background thread."""
        try:
            import pyaudio
            import numpy as np
            from openwakeword.model import Model

            # Load the wake word model
            if self._model is None:
                self._model = Model(
                    wakeword_models=["hey_jarvis"],
                    inference_framework="onnx",
                )

            # Open audio stream
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )

            logger.info("Wake word listener active — say 'Hey Jarvis'")

            while not self._stop_event.is_set():
                try:
                    audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)

                    # Run prediction
                    prediction = self._model.predict(audio_array)

                    # Check all wake word models
                    for model_name, score in prediction.items():
                        if score > self.threshold:
                            logger.info("Wake word detected! (model: %s, score: %.3f)", model_name, score)
                            self._model.reset()
                            if self.on_wake:
                                self.on_wake()
                            break

                except IOError:
                    continue
                except Exception as e:
                    logger.error("Wake word audio error: %s", e)
                    break

            stream.stop_stream()
            stream.close()
            pa.terminate()

        except ImportError as e:
            logger.error("Wake word dependencies missing: %s", e)
            self._available = False
        except Exception as e:
            logger.error("Wake word listener error: %s", e)
        finally:
            self.enabled = False
