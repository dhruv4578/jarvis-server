import json
import logging
import re
from typing import Any, Literal
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("jarvis.intent_router")

ActionType = Literal["CHAT", "OPEN_APP", "SET_TIMER", "ADD_NOTE"]


class HistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str


class IntentResponse(BaseModel):
    reply: str
    action: ActionType = "CHAT"
    target: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


SYSTEM_PROMPT = """You are JARVIS, a crisp, highly capable AI assistant powering a sci-fi HUD terminal.
Analyze the user's latest message and recent conversation history, determine the user's intent, and return ONLY a valid JSON object.

Output schema:
{
  "reply": "string (concise, natural assistant response in character)",
  "action": "CHAT" | "OPEN_APP" | "SET_TIMER" | "ADD_NOTE",
  "target": "string or null",
  "payload": {}
}

Supported Actions & Payload specifications:
1. "OPEN_APP":
   - Use when the user requests launching or opening an application, tool, or website (e.g., Spotify, Chrome, Camera, YouTube).
   - "target": Target application name (e.g., "Spotify").
   - "payload": {"app_name": "<name>"}

2. "SET_TIMER":
   - Use when the user requests setting a timer, alarm, or countdown.
   - "target": Concise summary or label (e.g., "10 minutes", "Pasta timer").
   - "payload": {"duration_seconds": <integer or null>, "label": "<string>"}

3. "ADD_NOTE":
   - Use when the user requests creating, saving, or remembering a note.
   - "target": Brief title or subject of the note, or null if unspecified.
   - "payload": {"text": "<content of note>"}

4. "CHAT":
   - Use for general conversation, answering questions, or when no physical HUD action is required.
   - "target": null
   - "payload": {}

Strict Rules:
- Return raw JSON only.
- Do NOT wrap your output in markdown code fences (no ```json).
- Do NOT output preamble or commentary.
"""


class IntentRouter:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-3.5-flash-lite",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def _call_llm(self, model: str, messages: list[dict[str, str]]) -> str:
        try:
            # Primary attempt: Request structured JSON format
            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content or ""
        except Exception:
            # Fallback attempt: Standard completion in case endpoint doesn't support json_object
            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            return completion.choices[0].message.content or ""

    async def route(
        self,
        message: str,
        history: list[HistoryItem] | None = None,
    ) -> IntentResponse:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for item in history[-10:]:
                messages.append({"role": item.role, "content": item.content})
        messages.append({"role": "user", "content": message})

        # Try configured model first; if high-demand 503 or 404 occurs, gracefully fallback
        models_to_try = [self.model]
        fallback_model = "gemini-3.5-flash-lite"
        if fallback_model not in models_to_try:
            models_to_try.append(fallback_model)

        raw_content = ""
        last_error: Exception | None = None
        for current_model in models_to_try:
            try:
                raw_content = await self._call_llm(current_model, messages)
                if raw_content:
                    break
            except Exception as err:
                last_error = err
                logger.warning(
                    "Model %s call failed (%s). Attempting fallback...",
                    current_model,
                    err,
                )

        if not raw_content and last_error:
            logger.error("All LLM attempts failed: %s", last_error)
            return IntentResponse(
                reply=f"JARVIS cognitive core connection error: {last_error}",
                action="CHAT",
                target=None,
                payload={},
            )

        return self._parse_response(raw_content)

    def _parse_response(self, raw_content: str) -> IntentResponse:
        cleaned = raw_content.strip()

        # Strip markdown code fences if present (e.g. ```json ... ``` or ``` ... ```)
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        elif cleaned.startswith("```"):
            lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        data: dict[str, Any] | None = None

        # Attempt 1: Direct JSON parsing
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            pass

        # Attempt 2: Extract substring from first '{' to last '}'
        if data is None:
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                try:
                    candidate = cleaned[first_brace : last_brace + 1]
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        data = parsed
                except json.JSONDecodeError:
                    pass

        # Fallback on malformed or non-JSON response
        if data is None:
            fallback_reply = (
                cleaned
                if cleaned
                else "Command received, sir. Response formatting was ambiguous, defaulting to chat."
            )
            return IntentResponse(
                reply=fallback_reply,
                action="CHAT",
                target=None,
                payload={},
            )

        # Normalize Action
        action_raw = str(data.get("action", "CHAT")).strip().upper()
        valid_actions: set[ActionType] = {"CHAT", "OPEN_APP", "SET_TIMER", "ADD_NOTE"}
        action: ActionType = action_raw if action_raw in valid_actions else "CHAT"

        # Normalize Reply
        reply = str(data.get("reply") or "Acknowledged.")

        # Extract target
        raw_target = data.get("target")
        target = (
            str(raw_target).strip()
            if raw_target is not None and str(raw_target).strip()
            else None
        )

        # Normalize Payload based on Action
        raw_payload = data.get("payload")
        payload = raw_payload if isinstance(raw_payload, dict) else {}

        if action == "OPEN_APP":
            app_name = str(payload.get("app_name") or target or "Application").strip()
            target = app_name
            payload = {"app_name": app_name}

        elif action == "SET_TIMER":
            raw_duration = payload.get("duration_seconds")
            duration_seconds: int | None = None
            if raw_duration is not None:
                try:
                    duration_seconds = int(raw_duration)
                except (ValueError, TypeError):
                    duration_seconds = None

            label = str(payload.get("label") or target or "Timer").strip()
            target = label
            payload = {"duration_seconds": duration_seconds, "label": label}

        elif action == "ADD_NOTE":
            text = str(payload.get("text") or target or "").strip()
            payload = {"text": text}

        else:  # CHAT
            target = None
            payload = {}

        return IntentResponse(
            reply=reply,
            action=action,
            target=target,
            payload=payload,
        )

    async def close(self) -> None:
        await self.client.close()