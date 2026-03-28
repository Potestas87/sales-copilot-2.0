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
import re
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
        self._always_actionable_customer = os.getenv("ALWAYS_ACTIONABLE_CUSTOMER", "true").strip().lower() in {
            "1", "true", "yes", "on"
        }

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
        parsed = self._parse_response(raw_text, transcript)
        return self._ensure_actionable_result(parsed, transcript, conversation_turns or [])

    @staticmethod
    def _detect_offer_from_text(text: str) -> dict:
        """Extract partial offer terms from free text."""
        t = (text or "").lower()
        detected: dict = {}

        if "quarterly" in t:
            detected["quarterly"] = True

        term_match = re.search(r"\b(12|18|24)\s*months?\b", t)
        if term_match:
            detected["term_months"] = int(term_match.group(1))

        initial_match = re.search(r"\$?\s*(\d{2,4})\s*(?:initial|first)", t)
        if initial_match:
            detected["initial"] = int(initial_match.group(1))
        else:
            initial_match = re.search(r"initial[^$]{0,25}\$?\s*(\d{2,4})", t)
            if initial_match:
                detected["initial"] = int(initial_match.group(1))
        if not initial_match:
            first_match = re.search(r"first[^$]{0,25}\$?\s*(\d{2,4})", t)
            if first_match:
                detected["initial"] = int(first_match.group(1))

        bimonthly_match = re.search(r"\$?\s*(\d{2,4})\s*(?:bimonthly|every two months|follow[- ]?ups?|monthly)", t)
        if bimonthly_match:
            detected["bimonthly"] = int(bimonthly_match.group(1))
        else:
            bimonthly_match = re.search(r"(bimonthly|every two months|follow[- ]?ups?|monthly)[^$]{0,35}\$?\s*(\d{2,4})", t)
            if bimonthly_match:
                detected["bimonthly"] = int(bimonthly_match.group(2))

        # Known pair fallback (e.g., "$99 initial and $120 bimonthly")
        if "initial" in t and ("bimonthly" in t or "every two months" in t):
            dollars = [int(x) for x in re.findall(r"\$+\s*(\d{2,4})", t)]
            if len(dollars) >= 2:
                detected.setdefault("initial", dollars[0])
                detected.setdefault("bimonthly", dollars[1])

        return detected

    def _derive_offer_progress(self, conversation_turns: list[dict]) -> dict:
        """
        Infer current/best offer and concession count from salesperson turns.
        Baseline: 24 months, $175 initial, $150 bimonthly.
        """
        current = {"initial": 175, "bimonthly": 150, "term_months": 24, "quarterly": False}
        best = dict(current)
        rac_steps = 0

        for turn in conversation_turns:
            if turn.get("speaker") != "salesperson":
                continue
            text = str(turn.get("transcript", "") or "")
            if not text.strip():
                continue

            detected = self._detect_offer_from_text(text)
            if not detected:
                continue

            previous = dict(current)
            current.update(detected)

            improved = (
                current.get("initial", previous["initial"]) < previous["initial"]
                or current.get("bimonthly", previous["bimonthly"]) < previous["bimonthly"]
                or current.get("term_months", previous["term_months"]) < previous["term_months"]
                or (current.get("quarterly") and not previous.get("quarterly"))
            )
            if improved:
                rac_steps += 1

            best["initial"] = min(best["initial"], current.get("initial", best["initial"]))
            best["bimonthly"] = min(best["bimonthly"], current.get("bimonthly", best["bimonthly"]))
            best["term_months"] = min(best["term_months"], current.get("term_months", best["term_months"]))
            best["quarterly"] = bool(best.get("quarterly") or current.get("quarterly"))

        return {"current": current, "best": best, "rac_steps": rac_steps}

    def _next_rac_suggestion(self, progress: dict, transcript: str) -> str:
        """Generate deterministic next RAC step without regressing the deal."""
        best = progress["best"]
        rac_steps = int(progress["rac_steps"])
        t = (transcript or "").lower()
        price_cue = any(k in t for k in ("price", "cost", "expensive", "total", "how much", "monthly"))
        term_cue = any(k in t for k in ("month", "term", "contract", "long"))

        initial = best["initial"]
        bimonthly = best["bimonthly"]
        term = best["term_months"]

        if rac_steps >= 3:
            return (
                "Final option: I can offer quarterly service from here so we still keep protection in place. "
                "Would quarterly service solve the concern enough to move forward?"
            )

        # RAC 1: move initial to 99, keep bimonthly 150
        if rac_steps == 0:
            initial = min(initial, 99)
            bimonthly = min(bimonthly, 150)
        # RAC 2/3: improve either term or price path, never regress
        elif rac_steps == 1:
            if price_cue and bimonthly > 120:
                initial = min(initial, 99)
                bimonthly = min(bimonthly, 120)
            elif term_cue and term > 18:
                term = 18
            else:
                if bimonthly > 120:
                    initial = min(initial, 99)
                    bimonthly = min(bimonthly, 120)
                elif term > 18:
                    term = 18
        elif rac_steps == 2:
            if price_cue and bimonthly > 99:
                initial = min(initial, 59)
                bimonthly = min(bimonthly, 99)
            elif term_cue and term > 12:
                term = 12
            else:
                if bimonthly > 99:
                    initial = min(initial, 59)
                    bimonthly = min(bimonthly, 99)
                elif term > 12:
                    term = 12

        return (
            f"Here is the next option I can do: ${initial} initial, then ${bimonthly} every two months, "
            f"on a {term}-month term. If this solves the concern, we can get this locked in now."
        )

    def _ensure_actionable_result(self, result: dict, transcript: str, conversation_turns: list[dict]) -> dict:
        """Force an actionable fallback for non-empty customer transcripts when configured."""
        transcript_text = (transcript or "").strip()
        if not transcript_text:
            return result

        progress = self._derive_offer_progress(conversation_turns)

        # Hard business rules always apply.
        enforced = self._apply_business_rules(result, transcript_text, progress)
        if enforced is not None:
            return enforced

        if not self._always_actionable_customer:
            return result

        suggestion_type = str(result.get("type", "none") or "none")
        suggestion_text = str(result.get("suggestion", "") or "").strip()
        confidence = float(result.get("confidence", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        reasoning_short = str(result.get("reasoning_short", "") or "").strip()

        if suggestion_type != "none" and suggestion_text:
            return result

        # Deterministic RAC fallback uses what has already been offered.
        rac_fallback = self._next_rac_suggestion(progress, transcript_text)
        fallback = {
            "type": "question",
            "suggestion": rac_fallback,
            "reasoning_short": reasoning_short or "Fallback prompt applied when model returned no actionable guidance.",
            "confidence": max(confidence, 0.35),
        }
        return fallback

    @staticmethod
    def _contains_sub_12_month_term(text: str) -> bool:
        for match in re.finditer(r"\b(\d{1,2})\s*months?\b", text.lower()):
            if int(match.group(1)) < 12:
                return True
        return False

    def _apply_business_rules(self, result: dict, transcript: str, progress: dict) -> Optional[dict]:
        """Enforce Brooks-specific guardrails and high-priority talk-track overrides."""
        transcript_l = transcript.lower()
        suggestion = str(result.get("suggestion", "") or "").strip()
        suggestion_l = suggestion.lower()

        # Spouse/partner authority smokescreen pullback.
        if any(phrase in transcript_l for phrase in ("talk to my husband", "talk to my wife", "talk to my spouse")):
            return {
                "type": "objection",
                "suggestion": (
                    "Absolutely, that makes sense. Before you do, what would your husband/wife need to hear "
                    "to feel good about moving forward so I can help you get that answered now?"
                ),
                "reasoning_short": "Authority smokescreen detected; pull back into a concrete next-step question.",
                "confidence": max(float(result.get("confidence", 0.0) or 0.0), 0.75),
            }

        # Pricing/total question deterministic response.
        if any(token in transcript_l for token in ("how much", "total", "price", "cost", "monthly")):
            best = progress["best"]
            return {
                "type": "question",
                "suggestion": (
                    f"Great question. Right now the initial service is ${best['initial']} (normally $350), "
                    f"then it's ${best['bimonthly']} every two months. "
                    f"We start at {best['term_months']} months and if needed we can step down to 18, then 12 to find the best fit."
                ),
                "reasoning_short": "Pricing question answered with Brooks anchors and approved term ladder.",
                "confidence": max(float(result.get("confidence", 0.0) or 0.0), 0.8),
            }

        # Never suggest a term below 12 months.
        if self._contains_sub_12_month_term(suggestion_l):
            return {
                "type": "objection",
                "suggestion": (
                    "I hear you. The approved options are 24 months first, then 18, then 12 months if needed. "
                    "We don't offer terms below 12 months, but we can choose the best fit within that ladder."
                ),
                "reasoning_short": "Term floor enforcement: minimum allowed term is 12 months.",
                "confidence": max(float(result.get("confidence", 0.0) or 0.0), 0.8),
            }

        # If model suggests an offer, never allow regression versus best already offered.
        detected_offer = self._detect_offer_from_text(suggestion)
        if detected_offer:
            best = progress["best"]
            regress_initial = "initial" in detected_offer and detected_offer["initial"] > best["initial"]
            regress_bimonthly = "bimonthly" in detected_offer and detected_offer["bimonthly"] > best["bimonthly"]
            regress_term = "term_months" in detected_offer and detected_offer["term_months"] > best["term_months"]
            if regress_initial or regress_bimonthly or regress_term:
                return {
                    "type": "objection",
                    "suggestion": (
                        f"Let's stay at your best offered terms so far: ${best['initial']} initial, "
                        f"${best['bimonthly']} every two months, {best['term_months']} months. "
                        "Would getting this finalized today work for you?"
                    ),
                    "reasoning_short": "Prevented regressive offer above already-conceded terms.",
                    "confidence": max(float(result.get("confidence", 0.0) or 0.0), 0.75),
                }

        return None

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

    @staticmethod
    def _repair_common_json_issues(text: str) -> str:
        """
        Repair common non-JSON escapes seen in LLM output.
        Example: "\\_" or "\\-" are invalid in JSON strings.
        """
        # Keep valid escapes intact, strip the backslash from invalid ones.
        return re.sub(r'\\([^"\\/bfnrtu])', r"\1", text)

    def _parse_response(self, raw_text: str, original_transcript: str) -> dict:
        """
        Parse the LLM's JSON response into a clean dict.

        If parsing fails for any reason, return a safe fallback so the client
        always receives a valid response — never a server error from bad JSON.
        """
        try:
            # Extract the first balanced object; the model may add prose around JSON.
            json_text = self._extract_first_json_object(raw_text)
            parsed = None
            parse_error = None
            candidates = (
                json_text,
                json_text.replace("\\_", "_"),
                self._repair_common_json_issues(json_text),
            )
            for candidate in candidates:
                try:
                    parsed = json.loads(candidate)
                    break
                except json.JSONDecodeError as e:
                    parse_error = e
            if parsed is None:
                raise parse_error  # type: ignore[misc]

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
