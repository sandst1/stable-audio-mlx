"""
Stable Audio MLX Sampler - A keyboard-controlled sampler UI for audio generation.

Usage:
    python sampler.py

Controls:
    - White keys: a=C4, s=D4, d=E4, f=F4, g=G4, h=A4, j=B4, k=C5
    - Black keys: w=C#4, e=D#4, t=F#4, y=G#4, u=A#4
    - Hold key to play (loops), release to stop
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import numpy as np
import sounddevice as sd
import mlx.core as mx
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QSlider, QLabel, QPushButton,
    QLineEdit, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QKeyEvent, QPainter, QColor, QFont, QPen

from src.pipeline.pipeline import StableAudioPipeline


# Note names for display
NOTE_NAMES = ['C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5']


class GenerationThread(QThread):
    """Thread for running audio generation without blocking the UI."""
    finished = pyqtSignal(object)  # Emits the audio array
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, pipeline, prompt, seconds_total, steps=8, cfg_scale=6.0, sampler="euler"):
        super().__init__()
        self.pipeline = pipeline
        self.prompt = prompt
        self.seconds_total = seconds_total
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.sampler = sampler

    def run(self):
        try:
            self.progress.emit("Generating audio...")
            audio = self.pipeline.generate(
                prompt=self.prompt,
                negative_prompt="",
                steps=self.steps,
                cfg_scale=self.cfg_scale,
                seconds_total=self.seconds_total,
                seed=None,  # Random seed
                sampler=self.sampler
            )
            # Convert to numpy: audio shape is (1, 2, T)
            audio_arr = np.array(audio[0], dtype=np.float32)  # (2, T)
            self.finished.emit(audio_arr)
        except Exception as e:
            self.error.emit(str(e))


class AudioPlayer:
    """Handles real-time audio playback with looping from different start positions."""

    # Fade duration in samples (~5ms at 44100Hz - very short to avoid clicks)
    FADE_SAMPLES = 220

    def __init__(self, sample_rate=44100):
        self.audio_data = None  # (samples, 2) stereo
        self.sample_rate = sample_rate
        self.current_position = 0
        self.loop_start = 0
        self.is_playing = False
        self.is_fading_in = False
        self.is_fading_out = False
        self.fade_position = 0
        self.stream = None

    def load_audio(self, audio_array):
        """Load audio from pipeline output (2, T) -> (T, 2)."""
        self.audio_data = audio_array.T.copy()
        # Normalize to prevent clipping
        max_val = np.abs(self.audio_data).max()
        if max_val > 0.95:
            self.audio_data = self.audio_data * (0.95 / max_val)
        # Ensure stream is created
        self._ensure_stream()

    def play_from(self, note_index):
        """Start playback from position based on note (0-12)."""
        if self.audio_data is None:
            return
        # Calculate start position based on note
        total_samples = len(self.audio_data)
        self.loop_start = int((note_index / 12) * total_samples)
        self.current_position = self.loop_start
        # Start fade-in
        self.is_fading_in = True
        self.is_fading_out = False
        self.fade_position = 0
        self.is_playing = True

    def stop(self):
        """Stop playback with fade-out."""
        if self.is_playing and not self.is_fading_out:
            # Start fade-out instead of abrupt stop
            self.is_fading_out = True
            self.is_fading_in = False
            self.fade_position = 0

    def cleanup(self):
        """Clean up the audio stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _ensure_stream(self):
        """Ensure the audio stream is created and running."""
        if self.stream is None:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype='float32',
                callback=self._audio_callback,
                blocksize=1024
            )
            self.stream.start()

    def _audio_callback(self, outdata, frames, time, status):
        """Called by sounddevice to fill audio buffer."""
        if not self.is_playing or self.audio_data is None:
            outdata.fill(0)
            return

        total_samples = len(self.audio_data)

        # Fill buffer with audio data (handling looping)
        filled = 0
        while filled < frames:
            end_pos = self.current_position + (frames - filled)
            if end_pos >= total_samples:
                # Copy remaining samples before loop
                remaining = total_samples - self.current_position
                if remaining > 0:
                    outdata[filled:filled + remaining] = self.audio_data[self.current_position:]
                    filled += remaining
                # Loop back
                self.current_position = self.loop_start
            else:
                chunk = frames - filled
                outdata[filled:filled + chunk] = self.audio_data[self.current_position:self.current_position + chunk]
                self.current_position += chunk
                filled += chunk

        # Apply fade-in envelope
        if self.is_fading_in:
            for i in range(frames):
                if self.fade_position < self.FADE_SAMPLES:
                    # Linear fade from 0 to 1
                    gain = self.fade_position / self.FADE_SAMPLES
                    outdata[i] *= gain
                    self.fade_position += 1
                else:
                    # Fade-in complete
                    self.is_fading_in = False
                    break

        # Apply fade-out envelope
        if self.is_fading_out:
            for i in range(frames):
                if self.fade_position < self.FADE_SAMPLES:
                    # Linear fade from 1 to 0
                    gain = 1.0 - (self.fade_position / self.FADE_SAMPLES)
                    outdata[i] *= gain
                    self.fade_position += 1
                else:
                    # Fade-out complete, stop playback
                    outdata[i:].fill(0)
                    self.is_playing = False
                    self.is_fading_out = False
                    break


class KeyboardWidget(QWidget):
    """Visual piano keyboard widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(192)
        self.setMinimumWidth(400)
        self.active_notes = set()  # Set of active note indices

        # Key labels
        self.white_labels = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k']
        self.black_labels = ['w', 'e', '', 't', 'y', 'u', '']

    def set_active_notes(self, notes):
        """Update which notes are currently active."""
        self.active_notes = notes
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # White key dimensions
        white_key_width = width // 8
        white_key_height = height

        # Black key dimensions
        black_key_width = white_key_width * 0.6
        black_key_height = height * 0.6

        # Draw white keys
        white_notes = [0, 2, 4, 5, 7, 9, 11, 12]  # C, D, E, F, G, A, B, C
        for i, note in enumerate(white_notes):
            x = i * white_key_width
            is_active = note in self.active_notes

            # Key background
            if is_active:
                painter.setBrush(QColor(100, 180, 255))
            else:
                painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawRect(int(x), 0, int(white_key_width - 1), int(white_key_height - 1))

            # Note name
            painter.setPen(QColor(100, 100, 100))
            font = QFont()
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(int(x + white_key_width // 2 - 10), int(white_key_height - 25), NOTE_NAMES[note])

            # Key label
            painter.setPen(QColor(50, 50, 50))
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(int(x + white_key_width // 2 - 5), int(white_key_height - 8), self.white_labels[i])

        # Draw black keys
        black_positions = [0, 1, 3, 4, 5]  # After C, D, F, G, A
        black_notes = [1, 3, 6, 8, 10]  # C#, D#, F#, G#, A#
        black_labels_active = ['w', 'e', 't', 'y', 'u']

        for i, (pos, note) in enumerate(zip(black_positions, black_notes)):
            x = (pos + 1) * white_key_width - black_key_width / 2
            is_active = note in self.active_notes

            if is_active:
                painter.setBrush(QColor(80, 140, 200))
            else:
                painter.setBrush(QColor(30, 30, 30))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawRect(int(x), 0, int(black_key_width), int(black_key_height))

            # Key label
            painter.setPen(QColor(255, 255, 255))
            font = QFont()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(int(x + black_key_width // 2 - 4), int(black_key_height - 8), black_labels_active[i])


class SamplerWindow(QMainWindow):
    """Main window for the sampler application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stable Audio MLX Sampler")
        self.setMinimumSize(500, 400)

        # State
        self.pipeline = None
        self.audio_player = AudioPlayer()
        self.active_keys = set()
        self.generation_thread = None
        self.current_prompt = ""

        # Key mapping: keyboard key -> note index (0-12)
        self.key_to_note = {
            Qt.Key.Key_A: 0,   # C4
            Qt.Key.Key_W: 1,   # C#4
            Qt.Key.Key_S: 2,   # D4
            Qt.Key.Key_E: 3,   # D#4
            Qt.Key.Key_D: 4,   # E4
            Qt.Key.Key_F: 5,   # F4
            Qt.Key.Key_T: 6,   # F#4
            Qt.Key.Key_G: 7,   # G4
            Qt.Key.Key_Y: 8,   # G#4
            Qt.Key.Key_H: 9,   # A4
            Qt.Key.Key_U: 10,  # A#4
            Qt.Key.Key_J: 11,  # B4
            Qt.Key.Key_K: 12,  # C5
        }

        self._setup_ui()
        self._load_pipeline()

    def _setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # BPM slider row
        bpm_layout = QHBoxLayout()
        bpm_label = QLabel("BPM:")
        self.bpm_slider = QSlider(Qt.Orientation.Horizontal)
        self.bpm_slider.setMinimum(60)
        self.bpm_slider.setMaximum(200)
        self.bpm_slider.setValue(120)
        self.bpm_value_label = QLabel("120")
        self.bpm_slider.valueChanged.connect(lambda v: self.bpm_value_label.setText(str(v)))
        bpm_layout.addWidget(bpm_label)
        bpm_layout.addWidget(self.bpm_slider, 1)
        bpm_layout.addWidget(self.bpm_value_label)
        layout.addLayout(bpm_layout)

        # Duration slider row
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Duration:")
        self.duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.duration_slider.setMinimum(2)
        self.duration_slider.setMaximum(10)
        self.duration_slider.setValue(5)
        self.duration_value_label = QLabel("5s")
        self.duration_slider.valueChanged.connect(lambda v: self.duration_value_label.setText(f"{v}s"))
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_slider, 1)
        duration_layout.addWidget(self.duration_value_label)
        layout.addLayout(duration_layout)

        # Prompt input row
        prompt_layout = QHBoxLayout()
        prompt_label = QLabel("Prompt:")
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("e.g., house drums with bass")
        self.prompt_input.returnPressed.connect(self._on_generate)
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_input, 1)
        layout.addLayout(prompt_layout)

        # Generate button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setMinimumWidth(120)
        self.generate_btn.clicked.connect(self._on_generate)
        button_layout.addWidget(self.generate_btn)
        layout.addLayout(button_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # Active sound label
        self.active_sound_label = QLabel("Active Sound: (none)")
        self.active_sound_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.active_sound_label)

        # Status label
        self.status_label = QLabel("Loading pipeline...")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        # Keyboard widget
        self.keyboard_widget = KeyboardWidget()
        layout.addWidget(self.keyboard_widget)

        # Instructions
        instructions = QLabel("Hold keys to play: a s d f g h j k (white) | w e t y u (black)")
        instructions.setStyleSheet("color: gray; font-size: 11px;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

    def _load_pipeline(self):
        """Load the pipeline in a background thread."""
        def load():
            weights_path = "model/stable_audio_small.npz"
            if not os.path.exists(weights_path):
                return None
            return StableAudioPipeline.from_pretrained(weights_path)

        class LoadThread(QThread):
            finished = pyqtSignal(object)

            def run(self):
                pipe = load()
                self.finished.emit(pipe)

        self.load_thread = LoadThread()
        self.load_thread.finished.connect(self._on_pipeline_loaded)
        self.load_thread.start()

    def _on_pipeline_loaded(self, pipeline):
        """Called when pipeline is loaded."""
        if pipeline is None:
            self.status_label.setText("Error: Model weights not found. Run src/conversion/convert.py first.")
            self.generate_btn.setEnabled(False)
        else:
            self.pipeline = pipeline
            self.status_label.setText("Ready - Enter a prompt and click Generate")

    def _on_generate(self):
        """Handle generate button click."""
        if self.pipeline is None:
            return

        prompt_text = self.prompt_input.text().strip()
        if not prompt_text:
            self.status_label.setText("Please enter a prompt")
            return

        # Build final prompt with BPM
        bpm = self.bpm_slider.value()
        full_prompt = f"{bpm}BPM {prompt_text}"
        self.current_prompt = prompt_text

        # Get duration
        duration = self.duration_slider.value()

        # Disable UI during generation
        self.generate_btn.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.status_label.setText(f"Generating: {full_prompt}...")

        # Start generation thread
        self.generation_thread = GenerationThread(
            self.pipeline,
            prompt=full_prompt,
            seconds_total=duration,
            steps=8,
            cfg_scale=6.0,
            sampler="euler"
        )
        self.generation_thread.finished.connect(self._on_generation_finished)
        self.generation_thread.error.connect(self._on_generation_error)
        self.generation_thread.start()

    def _on_generation_finished(self, audio_array):
        """Called when generation is complete."""
        self.audio_player.load_audio(audio_array)
        self.active_sound_label.setText(f"Active Sound: {self.current_prompt}")
        self.status_label.setText("Ready - Use keyboard to play")
        self.generate_btn.setEnabled(True)
        self.prompt_input.setEnabled(True)

    def _on_generation_error(self, error_msg):
        """Called when generation fails."""
        self.status_label.setText(f"Error: {error_msg}")
        self.generate_btn.setEnabled(True)
        self.prompt_input.setEnabled(True)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press for playing notes."""
        if event.isAutoRepeat():
            return

        key = event.key()
        if key in self.key_to_note and self.audio_player.audio_data is not None:
            note = self.key_to_note[key]
            self.active_keys.add(key)
            self.audio_player.play_from(note)
            self.status_label.setText(f"Playing: {NOTE_NAMES[note]}")
            # Update keyboard visualization
            active_notes = {self.key_to_note[k] for k in self.active_keys}
            self.keyboard_widget.set_active_notes(active_notes)

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release to stop playback."""
        if event.isAutoRepeat():
            return

        key = event.key()
        if key in self.active_keys:
            self.active_keys.discard(key)
            if not self.active_keys:
                self.audio_player.stop()
                self.status_label.setText("Ready")
            # Update keyboard visualization
            active_notes = {self.key_to_note[k] for k in self.active_keys}
            self.keyboard_widget.set_active_notes(active_notes)

    def closeEvent(self, event):
        """Clean up on close."""
        self.audio_player.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for consistent cross-platform look

    window = SamplerWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
