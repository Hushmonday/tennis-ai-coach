import json
import os
import re
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# ----------------------------
# Config
# ----------------------------

DEFAULT_MODEL_ID = os.getenv("NOVA_MODEL_ID", "amazon.nova-lite-v1:0")
DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")

# Increase timeout because LLM calls can take longer
_bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=DEFAULT_REGION,
    config=Config(
        connect_timeout=3600,
        read_timeout=3600,
        retries={"max_attempts": 1},
    ),
)

# ----------------------------
# System Prompt
# ----------------------------

SYSTEM_PROMPT = """You are a supportive professional tennis coach.

You will receive:
- shot_type: forehand | backhand | serve
- handedness: right | left
- camera_angle: side_view | semi_side_view | front_or_back | unknown
- metrics: numeric measurements
- flags: detected issues (strings)

Rules:
1) Start with 1 positive observation.
2) For each flag, explain why it matters in 1 sentence.
3) Give 1-2 actionable fixes per flag.
4) Provide a "Perfect / Pro-level standard" section using approximate ranges or qualitative targets.
5) End with encouragement.
6) Do NOT invent measurements.
7) Output JSON with keys: encouragement, issues, perfect_standard, closing.

Return only JSON.
"""

# ----------------------------
# Helper: Extract JSON from LLM output
# ----------------------------

def _extract_json(text: str) -> dict:
    """
    Extract JSON object from model output.
    Handles:
    - ```json ... ``` fenced blocks
    - extra text before/after JSON
    """
    if not text:
        raise ValueError("Empty model output")

    s = text.strip()

    # Remove fenced code block wrappers
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try extracting first {...} block
    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in output")

    return json.loads(match.group(0))


# ----------------------------
# Main Nova Call
# ----------------------------

def nova_coach_feedback(payload: dict, model_id: str = DEFAULT_MODEL_ID) -> dict:
    """
    Calls Amazon Nova via Bedrock Converse API and returns parsed JSON.
    """

    user_message = json.dumps(payload, ensure_ascii=False)

    messages = [
        {
            "role": "user",
            "content": [{"text": user_message}]
        }
    ]

    try:
        response = _bedrock_runtime.converse(
            modelId=model_id,
            messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={
                "maxTokens": 800,
                "temperature": 0.4,
                "topP": 0.9,
            },
        )

        text = response["output"]["message"]["content"][0]["text"]

    except (ClientError, Exception) as e:
        raise RuntimeError(f"Nova call failed: {e}")

    # Robust JSON parsing
    try:
        return _extract_json(text)
    except Exception:
        return {
            "encouragement": "Nice effort—your swing has a solid base.",
            "issues": [
                {
                    "flag": "parse_error",
                    "why_it_matters": "Model returned non-JSON output.",
                    "fixes": ["Try again."]
                }
            ],
            "perfect_standard": {},
            "closing": f"Raw output: {text[:500]}",
        }