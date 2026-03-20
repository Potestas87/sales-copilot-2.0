"""
display.py
----------
On-screen suggestion display — shows AI-generated responses during live calls.

Renders an always-on-top window that the salesperson can glance at while
talking. Updates in real-time as the server sends back suggestions.

Design goals:
  - Unobtrusive: small, positioned out of the way, semi-transparent
  - Scannable: colour-coded by type so you know what you're looking at
  - Non-blocking: UI runs on its own thread, never freezes the main app

Colour coding:
  Red    — objection    (needs a rebuttal)
  Blue   — question     (needs an answer)
  Green  — buying signal (keep momentum)
  Grey   — transcript only (no action needed)
"""

import queue
import threading
import tkinter as tk
from typing import Optional

# Colour scheme for each suggestion type
TYPE_COLOURS = {
    "objection":     {"bg": "#ff4444", "fg": "white",  "label": "OBJECTION"},
    "question":      {"bg": "#2196F3", "fg": "white",  "label": "QUESTION"},
    "buying_signal": {"bg": "#4CAF50", "fg": "white",  "label": "BUYING SIGNAL"},
    "none":          {"bg": "#424242", "fg": "#aaaaaa", "label": "TRANSCRIPT"},
}

WINDOW_WIDTH  = 480
WINDOW_HEIGHT = 200
WINDOW_X      = 20     # Distance from right edge of screen
WINDOW_Y      = 20     # Distance from top of screen


class SuggestionDisplay:
    """
    Tkinter-based always-on-top window that shows the latest suggestion.

    Tkinter must run on the main thread (macOS restriction), so this class
    manages its own internal update queue — other threads post updates via
    show(), and the tkinter mainloop drains them safely.

    Usage:
        display = SuggestionDisplay()
        display.start()                 # Blocks — call from main thread

        # From any other thread:
        display.show("suggestion text", "objection")
        display.stop()
    """

    def __init__(self):
        self._update_queue: queue.Queue = queue.Queue()
        self._root: Optional[tk.Tk]     = None
        self._running = False

    def show(self, suggestion: str, suggestion_type: str, transcript: str = "") -> None:
        """
        Queue a display update. Thread-safe — call from any thread.

        Args:
            suggestion:      The suggested response text (empty string for type "none")
            suggestion_type: One of: objection | question | buying_signal | none
            transcript:      The raw customer transcript (always shown)
        """
        self._update_queue.put({
            "suggestion": suggestion,
            "type":       suggestion_type,
            "transcript": transcript,
        })

    def start(self) -> None:
        """
        Build and start the tkinter window. Blocks until stop() is called.
        Must be called from the main thread on macOS.
        """
        self._running = True
        self._build_window()

        # Poll the update queue every 100ms using tkinter's after() scheduler.
        # This is the standard safe pattern for updating tkinter from other threads —
        # we never call tkinter methods directly from background threads.
        self._root.after(100, self._poll_updates)
        self._root.mainloop()

    def stop(self) -> None:
        """Signal the window to close."""
        self._running = False
        if self._root:
            self._root.quit()

    # ── Window construction ────────────────────────────────────────────────────
    def _build_window(self) -> None:
        """Create and configure the tkinter window."""
        self._root = tk.Tk()
        self._root.title("Sales Copilot")
        self._root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{WINDOW_X}+{WINDOW_Y}")

        # Always on top — stays visible even when the call app is in focus
        self._root.attributes("-topmost", True)

        # Semi-transparent background
        self._root.attributes("-alpha", 0.92)
        self._root.configure(bg="#1e1e1e")

        # ── Type badge (OBJECTION / QUESTION / etc.) ───────────────────────────
        self._badge = tk.Label(
            self._root,
            text       = "LISTENING...",
            font       = ("Helvetica Neue", 10, "bold"),
            bg         = "#424242",
            fg         = "white",
            padx       = 8,
            pady       = 4,
            anchor     = "w",
        )
        self._badge.pack(fill="x", padx=10, pady=(10, 4))

        # ── Suggestion text ────────────────────────────────────────────────────
        self._suggestion_label = tk.Label(
            self._root,
            text       = "Waiting for customer speech...",
            font       = ("Helvetica Neue", 13),
            bg         = "#1e1e1e",
            fg         = "#ffffff",
            wraplength = WINDOW_WIDTH - 24,
            justify    = "left",
            anchor     = "nw",
        )
        self._suggestion_label.pack(fill="both", expand=True, padx=12, pady=4)

        # ── Transcript (smaller, dimmer) ───────────────────────────────────────
        self._transcript_label = tk.Label(
            self._root,
            text       = "",
            font       = ("Helvetica Neue", 10),
            bg         = "#1e1e1e",
            fg         = "#777777",
            wraplength = WINDOW_WIDTH - 24,
            justify    = "left",
            anchor     = "nw",
        )
        self._transcript_label.pack(fill="x", padx=12, pady=(0, 10))

    # ── Update loop ────────────────────────────────────────────────────────────
    def _poll_updates(self) -> None:
        """
        Drain the update queue and refresh the UI.
        Called by tkinter's event loop every 100ms via after().
        """
        try:
            while True:
                update = self._update_queue.get_nowait()
                self._apply_update(update)
        except queue.Empty:
            pass

        if self._running:
            self._root.after(100, self._poll_updates)

    def _apply_update(self, update: dict) -> None:
        """Apply a queued update to the tkinter widgets."""
        suggestion_type = update.get("type", "none")
        suggestion      = update.get("suggestion", "")
        transcript      = update.get("transcript", "")

        colours = TYPE_COLOURS.get(suggestion_type, TYPE_COLOURS["none"])

        # Update badge
        self._badge.config(
            text = colours["label"],
            bg   = colours["bg"],
            fg   = colours["fg"],
        )

        # Update suggestion text
        if suggestion:
            self._suggestion_label.config(text=suggestion, fg="#ffffff")
        else:
            self._suggestion_label.config(
                text = "No action needed — continue listening",
                fg   = "#555555",
            )

        # Update transcript
        if transcript:
            # Truncate long transcripts to keep the window tidy
            display_transcript = transcript if len(transcript) <= 80 else transcript[:77] + "..."
            self._transcript_label.config(text=f'"{display_transcript}"')


# ── Standalone preview ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Preview the display UI with dummy data — no server or audio needed.
    Cycles through all suggestion types every 3 seconds.

    Usage:
        python3 client/display.py
    """
    import time

    display  = SuggestionDisplay()
    examples = [
        ("objection",     "That's a fair concern. Most customers feel that way initially — let me walk you through the ROI breakdown.",          "I'm not sure I can justify the cost right now."),
        ("question",      "Great question. The onboarding takes about two weeks and includes a dedicated manager the whole way through.",          "How long does it take to get set up?"),
        ("buying_signal", "Absolutely — we can have you live within the month. Want to pencil in a kickoff call for next week?",                   "This actually sounds like something we could really use."),
        ("none",          "",                                                                                                                      "We've been using our current setup for about three years."),
    ]

    def cycle_examples():
        for stype, suggestion, transcript in examples:
            display.show(suggestion, stype, transcript)
            time.sleep(3)
        display.stop()

    t = threading.Thread(target=cycle_examples, daemon=True)
    t.start()

    display.start()