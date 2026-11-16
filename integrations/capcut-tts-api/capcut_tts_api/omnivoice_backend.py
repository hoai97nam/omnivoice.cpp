"""
OmniVoice backend for CapCutClient.

Targets ``omnivoice-server`` (omnivoice.cpp), which serves both the OpenAI-style
``POST /v1/audio/speech`` endpoint and the CapCut common-task endpoints:

* ``POST /lv/v1/common_task/new``
* ``POST /lv/v1/common_task/query``
* ``GET /lv/v1/tts_artifact/<task_id>/<index>.<ext>``

Public ``CapCutClient`` TTS methods (``build_tts_new_request``,
``create_tts_task``, ``query_tts_task``, ``generate_speech``,
``extract_tts_artifacts``, ``download_tts_audio``, ``generate_speech_from_srt``,
``list_voices``, ``resolve_voice``) preserve their names, signatures, HTTP URIs,
and JSON response shapes across both ``backend="capcut"`` and
``backend="omnivoice"``.

STT / upload / translation are NOT provided by OmniVoice and raise
:class:`CapCutBackendNotSupportedError` (a ``CapCutError`` subclass).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.sax.saxutils import escape as _xml_escape

from capcut_tts_api.exceptions import (
    CapCutAPIError,
    CapCutBackendNotSupportedError,
    CapCutError,
)

DEFAULT_OMNIVOICE_URL = "http://localhost:8000"

_FEMALE_HINTS = ("female", "woman", "girl", "lady", "nữ", "chị", "cô", "女")
_MALE_HINTS = ("male", "man", "boy", "nam", "anh", "ông", "男")
_CHILD_HINTS = ("child", "kid", "bé", "童")
_ELDER_HINTS = ("elder", "elderly", "old", "già", "老")
_OMNIVOICE_ATTRIBUTES = (
    "male", "female", "child", "teenager", "young adult", "middle-aged", "elderly",
    "pitch", "whisper", "accent", "话",
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _load_json_option(value: Union[None, str, Path, Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Accept a dict, a JSON string, or a path to a JSON file."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    raw = str(value)
    path = Path(raw[1:] if raw.startswith("@") else raw)
    if not raw.lstrip().startswith("{") and path.exists():
        raw = path.read_text(encoding="utf-8-sig")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON object, JSON file path, or dict") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def capcut_lang_to_omnivoice(lang: Optional[str]) -> Optional[str]:
    """'vi-VN' -> 'vi', 'en-US' -> 'en', 'und'/'' -> None (auto-detect)."""
    if not lang:
        return None
    code = lang.strip().split("-")[0].split("_")[0].lower()
    if not code or code in ("und", "none", "auto"):
        return None
    return code


def _has_hint(text: str, words: set, hints: tuple) -> bool:
    """Match latin/Vietnamese hints as whole words, CJK hints as substrings."""
    for hint in hints:
        is_cjk = any(ord(c) >= 0x2E80 for c in hint)
        if (is_cjk and hint in text) or (not is_cjk and hint in words):
            return True
    return False


def _guess_instruct(*names: str) -> Optional[str]:
    """Best-effort voice-design prompt from a CapCut voice_type / display name."""
    text = " ".join(n for n in names if n).lower()
    words = set(re.split(r"[\s_\-.,()]+", text))
    parts: List[str] = []
    if _has_hint(text, words, _FEMALE_HINTS):
        parts.append("female")
    elif _has_hint(text, words, _MALE_HINTS):
        parts.append("male")
    if _has_hint(text, words, _CHILD_HINTS):
        parts.append("child")
    elif _has_hint(text, words, _ELDER_HINTS):
        parts.append("elderly")
    return ", ".join(parts) or None


def _looks_like_instruct(value: str) -> bool:
    lowered = value.lower()
    return any(attr in lowered for attr in _OMNIVOICE_ATTRIBUTES)


class OmniVoiceBackend:
    """HTTP adapter targeting omnivoice-server's CapCut common-task endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        session: Optional[Any] = None,
        timeout: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,
        response_format: Optional[str] = None,
        default_instruct: Optional[str] = None,
        default_language: Optional[str] = None,
        voice_map: Union[None, str, Path, Dict[str, Any]] = None,
        params: Union[None, str, Path, Dict[str, Any]] = None,
        max_workers: int = 1,
    ):
        del output_dir, max_workers  # kept in signature for backwards compatibility; tasks run on server
        self.base_url = (base_url or os.environ.get("OMNIVOICE_BASE_URL") or DEFAULT_OMNIVOICE_URL).rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("omnivoice base_url must be an absolute HTTP(S) URL")
        if session is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover
                raise CapCutError("The 'requests' package is required. Run 'pip install requests'.") from exc
            session = requests.Session()
        self.session = session
        self.timeout = float(timeout or _env_float("OMNIVOICE_TIMEOUT", 60.0))
        self.response_format = (response_format or os.environ.get("OMNIVOICE_RESPONSE_FORMAT") or "mp3").lower()
        if self.response_format not in ("mp3", "wav"):
            raise ValueError("OmniVoice response_format must be 'mp3' or 'wav'")
        self.default_instruct = default_instruct or os.environ.get("OMNIVOICE_DEFAULT_INSTRUCT") or None
        self.default_language = default_language or os.environ.get("OMNIVOICE_DEFAULT_LANGUAGE") or None
        self.voice_map = {
            str(k).lower(): v
            for k, v in _load_json_option(voice_map or os.environ.get("OMNIVOICE_VOICE_MAP"), "voice_map").items()
        }
        self.params = _load_json_option(params or os.environ.get("OMNIVOICE_PARAMS_JSON"), "params")
        self._presets: Optional[Dict[str, Dict[str, Any]]] = None

    # ------------------------------------------------------------------ server

    def health(self) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/health", timeout=10)
        if resp.status_code >= 400:
            raise CapCutAPIError(f"omnivoice health HTTP {resp.status_code}: {resp.text[:300]}", status_code=resp.status_code)
        return resp.json()

    def list_presets(self, refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Voice presets configured on the server via --voices-dir (name -> info)."""
        if self._presets is not None and not refresh:
            return self._presets
        presets: Dict[str, Dict[str, Any]] = {}
        try:
            resp = self.session.get(f"{self.base_url}/v1/audio/voices", timeout=10)
            if resp.status_code < 400:
                for item in (resp.json() or {}).get("data") or []:
                    if isinstance(item, dict) and item.get("id"):
                        presets[str(item["id"]).lower()] = item
        except Exception:
            presets = {}
        self._presets = presets
        return presets

    # ------------------------------------------------------- voice resolution

    def resolve_request_voice(
        self,
        voice: Optional[str],
        resource_id: Optional[str],
        catalog: List[Any],
    ) -> Dict[str, Any]:
        """Map a CapCut voice (voice_type / display name / resource_id) to OmniVoice fields.

        Priority:
        1. ``voice_map`` entry (key = voice_type, display name, or resource_id);
        2. server voice preset with the same name (voice cloning via --voices-dir);
        3. Voice.json catalog match -> language from ``lang`` + guessed voice design;
        4. unknown string that looks like an OmniVoice prompt ("female, low pitch") -> used as-is;
        5. defaults (OMNIVOICE_DEFAULT_INSTRUCT / OMNIVOICE_DEFAULT_LANGUAGE).
        """
        candidate = (voice or "").strip()
        lowered = candidate.lower()
        match = None
        for item in catalog:
            if lowered and (item.voice_type.lower() == lowered or item.display_name.lower() == lowered):
                match = item
                break
        if match is None and resource_id:
            match = next((item for item in catalog if item.resource_id == resource_id), None)

        keys = [lowered]
        if match is not None:
            keys += [match.voice_type.lower(), match.display_name.lower(), str(match.resource_id).lower()]
        if resource_id:
            keys.append(str(resource_id).lower())
        for key in keys:
            if key and key in self.voice_map:
                mapped = self.voice_map[key]
                if isinstance(mapped, str):
                    mapped = {"voice": mapped}
                fields = {k: v for k, v in dict(mapped).items() if v not in (None, "")}
                if "language" not in fields and match is not None:
                    lang = capcut_lang_to_omnivoice(match.lang) or self.default_language
                    if lang:
                        fields["language"] = lang
                return fields

        presets = self.list_presets()
        for key in keys:
            if key and key in presets:
                fields: Dict[str, Any] = {"voice": presets[key].get("id", key)}
                lang = capcut_lang_to_omnivoice(match.lang) if match is not None else None
                if lang or self.default_language:
                    fields["language"] = lang or self.default_language
                return fields

        fields = {}
        if match is not None:
            lang = capcut_lang_to_omnivoice(match.lang)
            instruct = _guess_instruct(match.voice_type, match.display_name) or self.default_instruct
        elif candidate and _looks_like_instruct(candidate):
            lang = None
            instruct = candidate
        else:
            lang = None
            instruct = self.default_instruct
        lang = lang or self.default_language
        if instruct:
            fields["instructions"] = instruct
        if lang:
            fields["language"] = lang
        return fields

    # -------------------------------------------------------------- requests

    def build_omnivoice_options(
        self,
        voice_fields: Dict[str, Any],
        rate: str = "1.0",
        language: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        opts: Dict[str, Any] = dict(self.params)
        opts.update(voice_fields)
        if language:
            mapped_lang = capcut_lang_to_omnivoice(language)
            if mapped_lang:
                opts["language"] = mapped_lang
        if instructions:
            opts["instructions"] = instructions
        if rate and "speed" not in opts:
            try:
                spd = float(rate)
                if spd > 0:
                    opts["speed"] = max(0.25, min(4.0, spd))
            except (TypeError, ValueError):
                pass
        opts["response_format"] = self.response_format
        return opts

    def build_tts_new_request(
        self,
        texts: List[str],
        voice_type: str,
        resource_id: str,
        voice_fields: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
        rate: str = "1.0",
        language: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Tuple[str, Dict[str, str], str]:
        """Build a CapCut-shaped POST /lv/v1/common_task/new request for omnivoice-server."""
        esc = lambda s: _xml_escape(s, {'"': "&quot;", "'": "&apos;"})
        rate_str = str(rate or "1.0")
        voice_blocks = [
            f'    <voice name="{esc(voice_type)}" mock_tone_info="" platform="sami" '
            f'resource_id="{esc(resource_id)}" emotion="" emotion_scale="0" style="" role="" '
            f'moyin_emotion="" is_clone_tone="false" need_subtitle_timestamp="false">\n'
            f'        <prosody rate="{esc(rate_str)}">{_xml_escape(text)}</prosody>\n'
            f'    </voice>'
            for text in texts
        ]
        ssml = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">\n'
            + "\n".join(voice_blocks)
            + "\n</speak>"
        )
        babi = {
            "feature_entrance": "editor",
            "feature_entrance_detail": "editor-feature-text_to_speech",
            "feature_key": "text_to_speech",
            "scenario": "video_editor",
        }
        payload = {
            "audio_format": self.response_format,
            "babi_param": json.dumps(babi, ensure_ascii=False, separators=(",", ":")),
            "credit_disable": False,
            "extra_info": json.dumps({"benefit_info": {}}, separators=(",", ":")),
            "need_merge_voice": False,
            "need_subtitle_timestamp": False,
            "scene": "text_to_speech",
            "ssml": ssml,
            "omnivoice": self.build_omnivoice_options(
                voice_fields, rate=rate_str, language=language, instructions=instructions
            ),
        }
        body = {
            "bind_id": str(uuid.uuid4()),
            "can_queue": True,
            "enter_from": "text_to_speech",
            "tasks": [
                {
                    "context": str(uuid.uuid4()),
                    "payload": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    "req_key": "sami_text_to_speech",
                    "task_version": "v3",
                }
            ],
        }
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if extra_headers:
            headers.update(extra_headers)
        return (
            f"{self.base_url}/lv/v1/common_task/new",
            headers,
            json.dumps(body, ensure_ascii=False),
        )

    def build_query_request(
        self,
        task_id: str,
        token: str,
        bind_id: str = "",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, str], str]:
        """Build a CapCut-shaped POST /lv/v1/common_task/query request for omnivoice-server."""
        body = {
            "tasks": [
                {
                    "bind_id": bind_id,
                    "id": task_id,
                    "req_key": "sami_text_to_speech",
                    "task_version": "v3",
                    "token": token,
                }
            ]
        }
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if extra_headers:
            headers.update(extra_headers)
        return (
            f"{self.base_url}/lv/v1/common_task/query",
            headers,
            json.dumps(body, ensure_ascii=False),
        )

    # ---------------------------------------------------------- unsupported

    @staticmethod
    def unsupported(feature: str) -> CapCutBackendNotSupportedError:
        return CapCutBackendNotSupportedError(
            f"{feature} is not supported by the 'omnivoice' backend (TTS only). "
            f"Use CapCutClient(backend='capcut') for speech-to-text, upload, and translation."
        )
