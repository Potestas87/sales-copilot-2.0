"""
inference.py
------------
Loads Mistral 7B and analyses customer utterances for objections, questions,
and buying signals — then generates a suggested response.

Why Mistral 7B?
  - Small enough to run on a single mid-range GPU (RTX 3090/4090, A4000)
  - Fast inference: ~1-2 seconds for a short response at 4-bit quantisation
  - High quality instruction-following — critical for structured JSON output
  - Open weights — no API costs, no rate limits, full control

Why GGUF + llama-cpp-python?
  GGUF is a quantisation format (created by llama.cpp) that stores model weights
  in 4-bit or 8-bit precision. This shrinks Mistral 7B from ~14GB (float16)
  down to ~4.5GB (Q4_K_M) — fitting comfortably on a 10GB GPU alongside Whisper.
  llama-cpp-python is Python bindings for llama.cpp, the fastest CPU/GPU inference
  engine for GGUF models.

Classification types:
  objection      — customer pushes back ("I'm not sure", "that's too expensive")
  question       — customer asks for information ("how does X work?")
  buying_signal  — customer shows interest ("when can I start?", "what's included?")
  none           — statement that doesn't need a response prompt
"""

import json
import logging
import os
from typing import Optional
from dotenv import load_dotenv
from llama_cpp import Llama

from prompts import build_system_prompt, build_user_prompt

load_dotenv()
log = logging.getLogger("inference")


class SuggestionEngine:
    """
    Loads Mistral 7B and generates sales suggestions from customer utterances.

    The engine:
      1. Classifies the utterance (objection / question / buying_signal / none)
      2. Uses recent conversation turns as context
      3. If actionable, generates a suggested response aligned with the sales playbook
      4. Returns a structured dict the API can send directly back to the Mac client

    Usage:
        engine = SuggestionEngine()
        result = engine.analyse("I'm not sure I can afford this right now")
        # result = {
        #   "type": "objection",
        #   "suggestion": "That's a valid concern. Let me show you the ROI breakdown..."
        # }
    """

    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        max_tokens      = int(os.getenv("LLM_MAX_TOKENS", 300))

        log.info(f"Loading LLM from '{self.model_name}'...")

        # n_gpu_layers=-1 offloads all layers to GPU — maximum speed.
        # n_ctx=2048 is the context window — enough for a sales conversation history.
        # verbose=False suppresses llama.cpp's internal logging noise.
        self._llm = Llama(
            model_path   = self.model_name,
            n_gpu_layers = -1,
            n_ctx        = 2048,
            verbose      = False,
        )
        self._max_tokens   = max_tokens
        self._system_prompt = build_system_prompt()

        log.info("LLM loaded.")

    def analyse(self, transcript: str, conversation_turns: Optional[list[dict]] = None) -> dict:
        """
        Classify and respond to a customer utterance.

        Args:
            transcript: Text of what the customer just said.
            conversation_turns: Recent dialogue turns (speaker + transcript).

        Returns:
            dict with keys:
              "type"       — one of: objection | question | buying_signal | none
              "suggestion" — suggested response string (empty string if type is "none")
              "reasoning_short" — one-sentence rationale for the recommendation
              "confidence" — model confidence estimate in [0.0, 1.0] when available

        The LLM is prompted to respond in JSON so we can parse it reliably.
        If the response can't be parsed, we fall back to a safe default.
        """
        messages = [
            {"role": "system",    "content": self._system_prompt},
            {"role": "user",      "content": build_user_prompt(transcript, conversation_turns)},
        ]

        log.info(f"Running inference on: '{transcript}'")

        response = self._llm.create_chat_completion(
            messages    = messages,
            max_tokens  = self._max_tokens,
            temperature = 0.3,    # Low temperature = more consistent, focused output
                                  # High temperature = more creative but less reliable
        )

        raw_text = response["choices"][0]["message"]["content"].strip()

        return self._parse_response(raw_text, transcript)

    def _extract_first_json_object(self, text: str) -> str:
        """Return the first balanced JSON object found in text."""
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in response")

        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        raise ValueError("Unterminated JSON object in response")

    def _parse_response(self, raw_text: str, original_transcript: str) -> dict:
        """
        Parse the LLM's JSON response into a clean dict.

        If parsing fails for any reason, return a safe fallback so the client
        always receives a valid response — never a server error from bad JSON.
        """
        try:
            # Extract the first balanced object; the model may add prose around JSON.
            json_text = self._extract_first_json_object(raw_text)
            parsed = json.loads(json_text)

            # Accept either direct schema or wrapped {"response": {...}} schema.
            if isinstance(parsed.get("response"), dict):
                parsed = parsed["response"]

            raw_type = parsed.get("type")
            raw_intent = parsed.get("intent")
            had_explicit_label = raw_type is not None or raw_intent is not None

            original_label = str(raw_type or raw_intent or "none").strip().lower()
            suggestion_type = original_label
            suggestion_type = suggestion_type.replace(" ", "_")
            if suggestion_type in {"concern", "pushback", "rebuttal"}:
                suggestion_type = "objection"
            elif suggestion_type in {"buying", "buying_intent", "buy_signal", "signal"}:
                suggestion_type = "buying_signal"

            confidence = float(parsed.get("confidence", 0.0) or 0.0)
            confidence = max(0.0, min(1.0, confidence))
            reasoning_short = str(
                parsed.get("reasoning_short")
                or parsed.get("reason")
                or ""
            ).strip()
            if len(reasoning_short) > 140:
                reasoning_short = reasoning_short[:140].rstrip()

            suggestion_text = str(
                parsed.get("suggestion")
                or parsed.get("message")
                or parsed.get("action")
                or ""
            ).strip()

            if suggestion_type not in {"objection", "question", "buying_signal", "none"}:
                # Unknown labels like "action"/"recommendation" should stay actionable
                # when we do have a usable suggestion text.
                if suggestion_text:
                    log.info("Normalising unknown intent label '%s' to 'question'", original_label)
                    suggestion_type = "question"
                else:
                    suggestion_type = "none"

            # Some model variants omit type but still provide actionable text.
            if suggestion_text and suggestion_type == "none" and not had_explicit_label:
                suggestion_type = "question"

            return {
                "type":       suggestion_type,
                "suggestion": suggestion_text,
                "reasoning_short": reasoning_short,
                "confidence": confidence,
            }

        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"Failed to parse LLM response: {e}. Raw: '{raw_text}'")
            return {
                "type":       "none",
                "suggestion": "",
                "reasoning_short": "",
                "confidence": 0.0,
            }
