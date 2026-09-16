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

# ── Palette ──────────────────────────────────────────────────────────────────
BG = "#121317"
CARD_BG = "#1b1d23"
CARD_BORDER = "#2c2f37"
DIVIDER = "#26282e"
TEXT_PRIMARY = "#f5f6f8"
TEXT_SECONDARY = "#9aa0ab"
TEXT_MUTED = "#6c7078"
ACCENT = "#6d93ff"

FONT_FAMILY = "Helvetica Neue"

# Color scheme for each suggestion type
TYPE_COLOURS = {
    "objection": {"bg": "#e5484d", "fg": "white", "label": "ACTION NOW · OBJECTION"},
    "question": {"bg": "#3b82f6", "fg": "white", "label": "ACTION NOW · QUESTION"},
    "buying_signal": {"bg": "#2fb344", "fg": "white", "label": "ACTION NOW · BUYING SIGNAL"},
    "none": {"bg": "#33363d", "fg": "#b8bcc4", "label": "LISTENING"},
}

# Color-coded badge (background + text) for the buying-temperature
# indicator — styled like TYPE_COLOURS above so it reads as a filled pill,
# not just tinted text, for a quick glance during a live call.
TEMPERATURE_COLOURS = {
    "hot": {"bg": "#e5484d", "fg": "white", "label": "\U0001F525  HOT"},
    "warm": {"bg": "#d99a3d", "fg": "white", "label": "WARM"},
    "cold": {"bg": "#3d72b4", "fg": "white", "label": "COLD"},
    "": {"bg": CARD_BG, "fg": TEXT_MUTED, "label": "Buying temp: —"},
}

WINDOW_WIDTH = 560
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

        # Persistent deal notepad — accumulates over the whole call, unlike
        # the suggestion card above which expires/clears per turn.
        self._notepad = {
            "customer_name": "",
            "address": "",
            "pain_points": [],
            "package_summary": "",
            "buying_temperature": "",
        }

    def show(
        self,
        suggestion: str,
        suggestion_type: str,
        transcript: str = "",
        speaker: str = "customer",
        confidence: float = 0.0,
        latency_ms: float = 0.0,
        reasoning_short: str = "",
        customer_name: str = "",
        address: str = "",
        pain_points: Optional[list] = None,
        package_summary: str = "",
        buying_temperature: str = "",
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
                "customer_name": customer_name,
                "address": address,
                "pain_points": pain_points or [],
                "package_summary": package_summary,
                "buying_temperature": buying_temperature,
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
        # Height is intentionally not fixed — the notepad (pain points in
        # particular) can grow across a call, and a hardcoded height either
        # clips that content or leaves a dead gap when it's short. Position
        # only; width is pinned via minsize + the wraplength values below,
        # height self-sizes to whatever's actually packed.
        self._root.geometry(f"+{WINDOW_X}+{WINDOW_Y}")
        self._root.minsize(WINDOW_WIDTH, 1)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", 0.96)
        self._root.configure(bg=BG)

        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self._root, bg=BG)
        header.pack(fill="x", padx=14, pady=(12, 8))

        tk.Label(
            header,
            text="●",
            font=(FONT_FAMILY, 9),
            bg=BG,
            fg=ACCENT,
        ).pack(side="left")

        tk.Label(
            header,
            text="  SALES COPILOT",
            font=(FONT_FAMILY, 11, "bold"),
            bg=BG,
            fg=ACCENT,
        ).pack(side="left")

        # ── Status badges ───────────────────────────────────────────────────
        self._badge = tk.Label(
            self._root,
            text="LISTENING",
            font=(FONT_FAMILY, 11, "bold"),
            bg=TYPE_COLOURS["none"]["bg"],
            fg=TYPE_COLOURS["none"]["fg"],
            padx=10,
            pady=6,
            anchor="w",
        )
        self._badge.pack(fill="x", padx=14, pady=(0, 3))

        self._temp_label = tk.Label(
            self._root,
            text=TEMPERATURE_COLOURS[""]["label"],
            font=(FONT_FAMILY, 9, "bold"),
            bg=TEMPERATURE_COLOURS[""]["bg"],
            fg=TEMPERATURE_COLOURS[""]["fg"],
            padx=10,
            pady=3,
            anchor="w",
        )
        self._temp_label.pack(fill="x", padx=14, pady=(0, 10))

        # ── Suggestion card ─────────────────────────────────────────────────
        suggestion_card = tk.Frame(self._root, bg=CARD_BG, highlightbackground=CARD_BORDER, highlightthickness=1)
        suggestion_card.pack(fill="x", padx=14, pady=(0, 10))

        self._suggestion_label = tk.Label(
            suggestion_card,
            text="Waiting for customer speech...",
            font=(FONT_FAMILY, 14),
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            wraplength=WINDOW_WIDTH - 56,
            justify="left",
            anchor="nw",
        )
        self._suggestion_label.pack(fill="x", padx=12, pady=(12, 6))

        self._meta_label = tk.Label(
            suggestion_card,
            text="",
            font=(FONT_FAMILY, 10),
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            wraplength=WINDOW_WIDTH - 56,
            justify="left",
            anchor="nw",
        )
        self._meta_label.pack(fill="x", padx=12, pady=(0, 12))

        # ── Live transcript ─────────────────────────────────────────────────
        transcript_frame = tk.Frame(self._root, bg=BG)
        transcript_frame.pack(fill="x", padx=14, pady=(0, 10))

        self._customer_label = tk.Label(
            transcript_frame,
            text="Customer:  (waiting)",
            font=(FONT_FAMILY, 10),
            bg=BG,
            fg=TEXT_PRIMARY,
            wraplength=WINDOW_WIDTH - 28,
            justify="left",
            anchor="nw",
        )
        self._customer_label.pack(fill="x", pady=(0, 3))

        self._sales_label = tk.Label(
            transcript_frame,
            text="You:  (waiting)",
            font=(FONT_FAMILY, 10),
            bg=BG,
            fg=TEXT_MUTED,
            wraplength=WINDOW_WIDTH - 28,
            justify="left",
            anchor="nw",
        )
        self._sales_label.pack(fill="x")

        # ── Persistent deal notepad ─────────────────────────────────────────
        notepad_frame = tk.Frame(self._root, bg=CARD_BG, highlightbackground=CARD_BORDER, highlightthickness=1)
        notepad_frame.pack(fill="x", padx=14, pady=(0, 14))

        notepad_title = tk.Label(
            notepad_frame,
            text="\U0001F4CB  DEAL NOTEPAD",
            font=(FONT_FAMILY, 10, "bold"),
            bg=CARD_BG,
            fg=ACCENT,
            anchor="w",
        )
        notepad_title.pack(fill="x", padx=12, pady=(10, 6))

        divider = tk.Frame(notepad_frame, bg=DIVIDER, height=1)
        divider.pack(fill="x", padx=12, pady=(0, 8))

        self._notepad_name_value = self._build_notepad_row(notepad_frame, "NAME")
        self._notepad_address_value = self._build_notepad_row(notepad_frame, "ADDRESS")
        self._notepad_package_value = self._build_notepad_row(notepad_frame, "PACKAGE")
        self._notepad_pain_value = self._build_notepad_row(notepad_frame, "PAIN POINTS", pady_bottom=10)

    def _build_notepad_row(self, parent: tk.Frame, field_label: str, pady_bottom: int = 4) -> tk.Label:
        """Build a "LABEL   value" row and return the value Label for later updates."""
        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(fill="x", padx=12, pady=(0, pady_bottom))

        tk.Label(
            row,
            text=field_label,
            font=(FONT_FAMILY, 8, "bold"),
            bg=CARD_BG,
            fg=TEXT_MUTED,
            width=11,
            anchor="nw",
        ).pack(side="left", anchor="n")

        value_label = tk.Label(
            row,
            text="—",
            font=(FONT_FAMILY, 10),
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            wraplength=WINDOW_WIDTH - 130,
            justify="left",
            anchor="nw",
        )
        value_label.pack(side="left", fill="x", expand=True)
        return value_label

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

        self._merge_notepad(update)
        self._render()

    def _merge_notepad(self, update: dict) -> None:
        """
        Overwrite-if-nonempty / append-dedup, mirroring the server's merge —
        belt-and-suspenders so a later message that omits a field never
        regresses a value the notepad already has.
        """
        new_name = str(update.get("customer_name", "") or "").strip()
        if new_name:
            self._notepad["customer_name"] = new_name

        new_address = str(update.get("address", "") or "").strip()
        if new_address:
            self._notepad["address"] = new_address

        new_package = str(update.get("package_summary", "") or "").strip()
        if new_package:
            self._notepad["package_summary"] = new_package

        new_temperature = str(update.get("buying_temperature", "") or "").strip().lower()
        if new_temperature in TEMPERATURE_COLOURS and new_temperature:
            self._notepad["buying_temperature"] = new_temperature

        for point in update.get("pain_points", []) or []:
            point = str(point or "").strip()
            if point and point not in self._notepad["pain_points"]:
                self._notepad["pain_points"].append(point)
        if len(self._notepad["pain_points"]) > 6:
            self._notepad["pain_points"] = self._notepad["pain_points"][-6:]

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

        temp = self._notepad.get("buying_temperature", "")
        temp_colours = TEMPERATURE_COLOURS.get(temp, TEMPERATURE_COLOURS[""])
        self._temp_label.config(text=temp_colours["label"], bg=temp_colours["bg"], fg=temp_colours["fg"])

        if self._current_suggestion:
            self._suggestion_label.config(
                text=self._clip_text(self._current_suggestion, 220),
                fg=TEXT_PRIMARY,
            )
            confidence_pct = int(round(self._current_confidence * 100))
            latency_label, latency_color = self._latency_visual(self._current_latency_ms)
            meta = f"Confidence {confidence_pct}%  ·  Latency {self._current_latency_ms:.0f}ms ({latency_label})"
            if self._current_reasoning_short:
                meta = f"{meta}\n{self._current_reasoning_short}"
            self._meta_label.config(text=meta, fg=latency_color)
        else:
            self._suggestion_label.config(
                text="No customer action needed right now.",
                fg=TEXT_MUTED,
            )
            self._meta_label.config(text="Listening for next customer turn...", fg=TEXT_MUTED)

        customer_text = self._latest_customer_text or "(waiting)"
        sales_text = self._latest_sales_text or "(waiting)"
        self._customer_label.config(text=f"Customer:  {customer_text}")
        self._sales_label.config(text=f"You:  {sales_text}")

        self._render_notepad()

    def _render_notepad(self) -> None:
        pain_points = ", ".join(self._notepad["pain_points"])

        self._notepad_name_value.config(text=self._notepad["customer_name"] or "—")
        self._notepad_address_value.config(text=self._notepad["address"] or "—")
        self._notepad_package_value.config(text=self._notepad["package_summary"] or "—")
        self._notepad_pain_value.config(text=self._clip_text(pain_points, 160) or "—")

    @staticmethod
    def _clip_text(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."

    @staticmethod
    def _latency_visual(latency_ms: float) -> tuple[str, str]:
        if latency_ms <= LATENCY_GOOD_MS:
            return "good", "#6fcf7d"
        if latency_ms <= LATENCY_WARN_MS:
            return "watch", "#d9a441"
        return "slow", "#e5646a"


if __name__ == "__main__":
    display = SuggestionDisplay()
    examples = [
        {
            "speaker": "customer",
            "type": "objection",
            "transcript": "I am not sure we can justify the cost right now. This is Sarah, by the way.",
            "suggestion": "Totally fair. Teams like yours usually justify this via reduced churn in under one quarter.",
            "reasoning_short": "Pricing concern detected; ROI framing is most relevant.",
            "confidence": 0.86,
            "latency_ms": 1240.0,
            "customer_name": "Sarah",
            "pain_points": ["worried about cost"],
            "buying_temperature": "cold",
        },
        {
            "speaker": "salesperson",
            "type": "none",
            "transcript": "I can do $99 initial and $150 bimonthly on a 24-month term.",
            "suggestion": "",
            "reasoning_short": "",
            "confidence": 0.0,
            "latency_ms": 980.0,
            "package_summary": "$99 initial, $150/2mo, 24-month term",
        },
        {
            "speaker": "customer",
            "type": "question",
            "transcript": "How long does onboarding usually take?",
            "suggestion": "Great question. Typical onboarding is two weeks with a dedicated implementation lead.",
            "reasoning_short": "Direct onboarding question; answer with timeline and support model.",
            "confidence": 0.91,
            "latency_ms": 1105.0,
            "pain_points": ["concerned about onboarding time"],
            "buying_temperature": "hot",
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
                customer_name=event.get("customer_name", ""),
                address=event.get("address", ""),
                pain_points=event.get("pain_points", []),
                package_summary=event.get("package_summary", ""),
                buying_temperature=event.get("buying_temperature", ""),
            )
            time.sleep(3)
        display.stop()

    t = threading.Thread(target=cycle_examples, daemon=True)
    t.start()
    display.start()
