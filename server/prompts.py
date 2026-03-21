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
      3. The tone to use in suggestions
      4. How to format its output (always JSON)
    """
    playbook = load_playbook()

    product_name  = playbook.get("product_name",  "our product")
    product_desc  = playbook.get("product_description", "")
    value_props   = playbook.get("value_propositions", [])
    tone          = playbook.get("tone", "professional, confident, and empathetic")

    value_prop_text = ""
    if value_props:
        value_prop_text = "Key value propositions:\n" + "\n".join(f"  - {v}" for v in value_props)

    system_prompt = f"""You are a real-time sales copilot assistant. Your job is to analyse what a customer
just said on a sales call and help the salesperson respond effectively.

Product: {product_name}
{product_desc}
{value_prop_text}

Tone: {tone}

Your task:
1. Classify what the customer said into one of these types:
   - "objection"     : customer pushes back, expresses doubt, or raises a concern
   - "question"      : customer asks for information or clarification
   - "buying_signal" : customer expresses interest, asks about next steps, or shows intent to buy
   - "none"          : general statement that doesn't require a specific sales response

2. If the type is objection, question, or buying_signal — write a short, natural suggested
   response the salesperson can use. Keep it under 3 sentences. Don't be robotic.

3. If the type is "none" — return an empty suggestion string.

IMPORTANT: Always respond in valid JSON using exactly this format:
{{"type": "<type>", "suggestion": "<suggestion text or empty string>"}}

Do not include any text outside the JSON object. Do not add explanation or commentary."""

    return system_prompt


def build_user_prompt(transcript: str) -> str:
    """
    Build the per-utterance user prompt sent alongside each transcription.

    Kept simple intentionally — the system prompt carries all the context.
    The user prompt just delivers the raw customer speech.
    """
    return f'Customer just said: "{transcript}"\n\nRespond with JSON only.'