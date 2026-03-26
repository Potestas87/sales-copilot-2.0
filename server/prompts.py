"""
prompts.py
----------
Builds the system and user prompts that are sent to the LLM.

Keeping prompts in their own module means:
  - You can update the sales playbook without touching inference logic
  - Prompts are easy to test and iterate on independently
  - The system prompt is built once at startup and reused, not rebuilt per-request
"""

import os
import yaml
import logging
from typing import Optional

log = logging.getLogger("prompts")

PLAYBOOK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "sales_playbook.yaml"
)


def load_playbook() -> dict:
    """
    Load the sales playbook from config/sales_playbook.yaml.

    The playbook contains:
      - product name and description
      - key value propositions
      - common objections and ideal responses
      - tone and style guidelines

    Returns an empty dict if the file doesn't exist yet — the system prompt
    will still work, just without company-specific context.
    """
    try:
        with open(PLAYBOOK_PATH, "r") as f:
            playbook = yaml.safe_load(f)
            log.info("Sales playbook loaded.")
            return playbook or {}
    except FileNotFoundError:
        log.warning(f"No playbook found at {PLAYBOOK_PATH}. Using generic prompt.")
        return {}


def build_system_prompt() -> str:
    """
    Build the system prompt that shapes the LLM's behaviour for the entire session.

    This is loaded once at startup and stays constant across all utterances.
    It tells the LLM:
      1. Its role (sales copilot, not a chatbot)
      2. What the product is and what makes it valuable
      3. How to handle specific objections (from playbook)
      4. How to respond to buying signals (from playbook)
      5. The tone to use in suggestions
      6. How to format its output (always JSON)
    """
    playbook = load_playbook()

    product_name  = playbook.get("product_name",  "our product")
    product_desc  = playbook.get("product_description", "")
    value_props   = playbook.get("value_propositions", [])
    tone          = playbook.get("tone", "professional, confident, and empathetic")
    objection_guidance  = playbook.get("objection_guidance", {})
    buying_signal_guide = playbook.get("buying_signal_guidance", "")

    value_prop_text = ""
    if value_props:
        value_prop_text = "Key value propositions:\n" + "\n".join(f"  - {v}" for v in value_props)

    # Build objection handling guidance from playbook
    objection_text = ""
    if objection_guidance:
        entries = []
        for key, info in objection_guidance.items():
            summary = info.get("summary", key)
            angle   = info.get("angle", "").strip()
            entries.append(f'  "{summary}" — {angle}')
        objection_text = (
            "Objection handling playbook (use these angles when the customer raises these concerns):\n"
            + "\n".join(entries)
        )

    buying_signal_text = ""
    if buying_signal_guide:
        buying_signal_text = f"Buying signal guidance:\n  {buying_signal_guide.strip()}"

    system_prompt = f"""You are a real-time sales copilot assistant. Your job is to analyse what a customer
just said on a sales call and help the salesperson respond effectively.

Product: {product_name}
{product_desc}
{value_prop_text}

{objection_text}

{buying_signal_text}

Tone: {tone}

Your task:
1. Classify what the customer said into one of these types:
   - "objection"     : customer pushes back, expresses doubt, raises a concern, mentions price,
                        says they're too busy, mentions a competitor, says it's complicated, or
                        says they need to check with someone else. ANY hesitation or pushback counts.
   - "question"      : customer asks for information, clarification, or wants to know more about
                        anything — features, pricing, process, timelines, integrations, etc.
   - "buying_signal" : customer expresses interest, asks about next steps, pricing details,
                        implementation, trials, demos, or says things like "how do we get started",
                        "what does onboarding look like", "can we try it", "who else uses this".
   - "none"          : ONLY use this for pure small talk, greetings, or completely off-topic
                        statements like "nice weather" or "hello". If in doubt, do NOT classify
                        as "none" — pick the closest actionable type instead.

2. If the type is objection, question, or buying_signal — write a short, natural suggested
   response the salesperson can use. Keep it under 3 sentences. Don't be robotic.
   Use the conversation context to avoid repeating what the salesperson already said.
   Add incremental value (new framing, evidence, or a concise next-step question).
   Draw from the objection handling playbook and value propositions above.

3. ONLY return "none" with an empty suggestion for pure small talk or greetings.
   Almost everything a customer says on a sales call is actionable — classify it.

4. Provide a short reason for the recommendation in one sentence (<= 140 chars)
   in a field called "reasoning_short".

5. Provide confidence in your classification and recommendation in a field called
   "confidence" as a float between 0.0 and 1.0.

IMPORTANT: Always respond in valid JSON using exactly this format:
{{"type": "<type>", "suggestion": "<suggestion text or empty string>", "reasoning_short": "<brief rationale>", "confidence": <0.0-1.0>}}

Do not include any text outside the JSON object. Do not add explanation or commentary."""

    return system_prompt


def build_user_prompt(transcript: str, conversation_turns: Optional[list[dict]] = None) -> str:
    """
    Build the per-utterance user prompt sent alongside each transcription.

    Includes recent conversation turns so the model can respond with context.
    """
    turns = conversation_turns or []

    conversation_block = "No prior conversation context."
    if turns:
        formatted = []
        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            text = (turn.get("transcript", "") or "").strip()
            if not text:
                continue
            formatted.append(f"{speaker}: {text}")
        if formatted:
            conversation_block = "\n".join(formatted)

    return (
        "Conversation so far (most recent turns):\n"
        f"{conversation_block}\n\n"
        f'Latest customer utterance: "{transcript}"\n\n'
        "Do not repeat prior salesperson statements; build on them.\n"
        "Respond with JSON only."
    )
