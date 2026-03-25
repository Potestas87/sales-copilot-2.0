"""
display.py
----------
On-screen suggestion display for live calls.

Shows:
  - Current actionable suggestion (customer turns only)
  - Confidence, latency, and short rationale
  - Most recent customer transcript line
  - Most recent salesperson transcript line
"""

import os
import queue
import threading
import time
import tkinter as tk
from typing import Optional

# Color scheme for each suggestion type
TYPE_COLOURS = {
    "objection": {"bg": "#ff4444", "fg": "white", "label": "ACTION NOW: OBJECTION"},
    "question": {"bg": "#2196F3", "fg": "white", "label": "ACTION NOW: QUESTION"},
    "buying_signal": {"bg": "#4CAF50", "fg": "white", "label": "ACTION NOW: BUYING SIGNAL"},
    "none": {"bg": "#424242", "fg": "#aaaaaa", "label": "LISTENING"},
}

WINDOW_WIDTH = 520
WINDOW_HEIGHT = 320
WINDOW_X = 20
WINDOW_Y = 20
STALE_SUGGESTION_SECONDS = float(os.getenv("STALE_SUGGESTION_SECONDS", 12))
LATENCY_GOOD_MS = float(os.getenv("LATENCY_GOOD_MS", 1500))
LATENCY_WARN_MS = float(os.getenv("LATENCY_WARN_MS", 2500))


class SuggestionDisplay:
    """Tkinter always-on-top window that shows turn-aware transcript and guidance."""

    def __init__(self):
        self._update_queue: queue.Queue = queue.Queue()
        self._root: Optional[tk.Tk] = None
        self._running = False

        self._latest_customer_text = ""
        self._latest_sales_text = ""
        self._current_suggestion = ""
        self._current_suggestion_type = "none"
        self._current_confidence = 0.0
        self._current_latency_ms = 0.0
        self._current_reasoning_short = ""
        self._last_actionable_ts = 0.0

    def show(
        self,
        suggestion: str,
        suggestion_type: str,
        transcript: str = "",
        speaker: str = "customer",
        confidence: float = 0.0,
        latency_ms: float = 0.0,
        reasoning_short: str = "",
    ) -> None:
        """Queue a display update. Thread-safe and non-blocking."""
        self._update_queue.put(
            {
                "suggestion": suggestion,
                "type": suggestion_type,
                "transcript": transcript,
                "speaker": speaker,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "reasoning_short": reasoning_short,
            }
        )

    def start(self) -> None:
        """Build and start the tkinter window. Blocks until stop() is called."""
        self._running = True
        self._build_window()
        self._root.after(100, self._poll_updates)
        self._root.mainloop()

    def stop(self) -> None:
        """Signal the window to close."""
        self._running = False
        if self._root:
            self._root.quit()

    def _build_window(self) -> None:
        self._root = tk.Tk()
        self._root.title("Sales Copilot")
        self._root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{WINDOW_X}+{WINDOW_Y}")
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", 0.94)
        self._root.configure(bg="#1e1e1e")

        self._badge = tk.Label(
            self._root,
            text="LISTENING",
            font=("Helvetica Neue", 10, "bold"),
            bg="#424242",
            fg="white",
            padx=8,
            pady=4,
            anchor="w",
        )
        self._badge.pack(fill="x", padx=10, pady=(10, 4))

        self._suggestion_label = tk.Label(
            self._root,
            text="Waiting for customer speech...",
            font=("Helvetica Neue", 13),
            bg="#1e1e1e",
            fg="#ffffff",
            wraplength=WINDOW_WIDTH - 24,
            justify="left",
            anchor="nw",
        )
        self._suggestion_label.pack(fill="x", padx=12, pady=(2, 4))

        self._meta_label = tk.Label(
            self._root,
            text="",
            font=("Helvetica Neue", 10),
            bg="#1e1e1e",
            fg="#9e9e9e",
            wraplength=WINDOW_WIDTH - 24,
            justify="left",
            anchor="nw",
        )
        self._meta_label.pack(fill="x", padx=12, pady=(0, 4))

        self._customer_label = tk.Label(
            self._root,
            text="Customer: (waiting)",
            font=("Helvetica Neue", 10),
            bg="#1e1e1e",
            fg="#d0d0d0",
            wraplength=WINDOW_WIDTH - 24,
            justify="left",
            anchor="nw",
        )
        self._customer_label.pack(fill="x", padx=12, pady=(4, 2))

        self._sales_label = tk.Label(
            self._root,
            text="You: (waiting)",
            font=("Helvetica Neue", 10),
            bg="#1e1e1e",
            fg="#9a9a9a",
            wraplength=WINDOW_WIDTH - 24,
            justify="left",
            anchor="nw",
        )
        self._sales_label.pack(fill="x", padx=12, pady=(0, 10))

    def _poll_updates(self) -> None:
        try:
            while True:
                update = self._update_queue.get_nowait()
                self._apply_update(update)
        except queue.Empty:
            pass

        self._expire_stale_suggestion()

        if self._running:
            self._root.after(100, self._poll_updates)

    def _apply_update(self, update: dict) -> None:
        suggestion_type = update.get("type", "none")
        suggestion = update.get("suggestion", "")
        transcript = update.get("transcript", "")
        speaker = update.get("speaker", "customer")
        confidence = float(update.get("confidence", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        latency_ms = float(update.get("latency_ms", 0.0) or 0.0)
        latency_ms = max(0.0, latency_ms)
        reasoning_short = str(update.get("reasoning_short", "") or "").strip()

        if transcript:
            clipped = self._clip_text(transcript, 110)
            if speaker == "customer":
                self._latest_customer_text = clipped
            elif speaker == "salesperson":
                self._latest_sales_text = clipped

        # Only customer turns can refresh actionable suggestions.
        if speaker == "customer":
            if suggestion and suggestion_type != "none":
                self._current_suggestion = suggestion
                self._current_suggestion_type = suggestion_type
                self._current_confidence = confidence
                self._current_latency_ms = latency_ms
                self._current_reasoning_short = self._clip_text(reasoning_short, 140)
                self._last_actionable_ts = time.monotonic()
            else:
                self._clear_actionable()

        self._render()

    def _expire_stale_suggestion(self) -> None:
        if self._current_suggestion_type == "none":
            return
        age = time.monotonic() - self._last_actionable_ts
        if age > STALE_SUGGESTION_SECONDS:
            self._clear_actionable()
            self._render()

    def _clear_actionable(self) -> None:
        self._current_suggestion = ""
        self._current_suggestion_type = "none"
        self._current_confidence = 0.0
        self._current_latency_ms = 0.0
        self._current_reasoning_short = ""

    def _render(self) -> None:
        colours = TYPE_COLOURS.get(self._current_suggestion_type, TYPE_COLOURS["none"])
        self._badge.config(text=colours["label"], bg=colours["bg"], fg=colours["fg"])

        if self._current_suggestion:
            self._suggestion_label.config(
                text=self._clip_text(self._current_suggestion, 220),
                fg="#ffffff",
            )
            confidence_pct = int(round(self._current_confidence * 100))
            latency_label, latency_color = self._latency_visual(self._current_latency_ms)
            meta = f"Confidence: {confidence_pct}% | Latency: {self._current_latency_ms:.0f}ms ({latency_label})"
            if self._current_reasoning_short:
                meta = f"{meta} | {self._current_reasoning_short}"
            self._meta_label.config(text=meta, fg=latency_color)
        else:
            self._suggestion_label.config(
                text="No customer action needed right now.",
                fg="#666666",
            )
            self._meta_label.config(text="Listening for next customer turn...", fg="#707070")

        customer_text = self._latest_customer_text or "(waiting)"
        sales_text = self._latest_sales_text or "(waiting)"
        self._customer_label.config(text=f"Customer: {customer_text}")
        self._sales_label.config(text=f"You: {sales_text}")

    @staticmethod
    def _clip_text(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."

    @staticmethod
    def _latency_visual(latency_ms: float) -> tuple[str, str]:
        if latency_ms <= LATENCY_GOOD_MS:
            return "good", "#7bc67b"
        if latency_ms <= LATENCY_WARN_MS:
            return "watch", "#e6c266"
        return "slow", "#e57373"


if __name__ == "__main__":
    display = SuggestionDisplay()
    examples = [
        {
            "speaker": "customer",
            "type": "objection",
            "transcript": "I am not sure we can justify the cost right now.",
            "suggestion": "Totally fair. Teams like yours usually justify this via reduced churn in under one quarter.",
            "reasoning_short": "Pricing concern detected; ROI framing is most relevant.",
            "confidence": 0.86,
            "latency_ms": 1240.0,
        },
        {
            "speaker": "salesperson",
            "type": "none",
            "transcript": "Would it help if I showed a 90-day ROI model?",
            "suggestion": "",
            "reasoning_short": "",
            "confidence": 0.0,
            "latency_ms": 980.0,
        },
        {
            "speaker": "customer",
            "type": "question",
            "transcript": "How long does onboarding usually take?",
            "suggestion": "Great question. Typical onboarding is two weeks with a dedicated implementation lead.",
            "reasoning_short": "Direct onboarding question; answer with timeline and support model.",
            "confidence": 0.91,
            "latency_ms": 1105.0,
        },
    ]

    def cycle_examples():
        for event in examples:
            display.show(
                suggestion=event["suggestion"],
                suggestion_type=event["type"],
                transcript=event["transcript"],
                speaker=event["speaker"],
                confidence=event["confidence"],
                latency_ms=event["latency_ms"],
                reasoning_short=event["reasoning_short"],
            )
            time.sleep(3)
        display.stop()

    t = threading.Thread(target=cycle_examples, daemon=True)
    t.start()
    display.start()
