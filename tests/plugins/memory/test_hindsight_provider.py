import asyncio
import json
import re

from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    _normalize_retain_tags,
)


class FakeRecallResponse:
    def __init__(self, results=None):
        self.results = results or []


class FakeReflectResponse:
    def __init__(self, text=""):
        self.text = text


class FakeClient:
    def __init__(self):
        self.retain_calls = []
        self.recall_calls = []
        self.reflect_calls = []
        self.closed = False

    async def aretain(self, **kwargs):
        self.retain_calls.append(kwargs)
        return {"ok": True}

    async def arecall(self, **kwargs):
        self.recall_calls.append(kwargs)
        return FakeRecallResponse()

    async def areflect(self, **kwargs):
        self.reflect_calls.append(kwargs)
        return FakeReflectResponse("ok")

    async def aclose(self):
        self.closed = True


def _make_provider(monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.hindsight._load_config",
        lambda: {
            "mode": "cloud",
            "api_url": "http://example.local",
            "bank_id": "josh-global",
            "budget": "high",
            "retain_tags": ["agent:abner", "source_system:hermes-agent"],
            "retain_source": "hermes",
            "retain_user_prefix": "User (Josh)",
            "retain_assistant_prefix": "Assistant (Abner)",
        },
    )
    monkeypatch.setattr("plugins.memory.hindsight._run_sync", lambda coro, timeout=120.0: asyncio.run(coro))
    provider = HindsightMemoryProvider()
    provider.initialize("session-1")
    provider._client = FakeClient()
    return provider


def test_normalize_retain_tags_accepts_csv_and_dedupes():
    assert _normalize_retain_tags("agent:abner, source_system:hermes-agent, agent:abner") == [
        "agent:abner",
        "source_system:hermes-agent",
    ]


def test_normalize_retain_tags_accepts_json_array_string():
    value = json.dumps(["agent:abner", "source_system:hermes-agent"])
    assert _normalize_retain_tags(value) == ["agent:abner", "source_system:hermes-agent"]


def test_initialize_loads_retain_config(monkeypatch):
    provider = _make_provider(monkeypatch)

    assert provider._retain_tags == ["agent:abner", "source_system:hermes-agent"]
    assert provider._retain_source == "hermes"
    assert provider._retain_user_prefix == "User (Josh)"
    assert provider._retain_assistant_prefix == "Assistant (Abner)"


def test_get_config_schema_exposes_retain_knobs():
    provider = HindsightMemoryProvider()
    keys = {field["key"] for field in provider.get_config_schema()}

    assert "retain_tags" in keys
    assert "retain_source" in keys
    assert "retain_user_prefix" in keys
    assert "retain_assistant_prefix" in keys


def test_sync_turn_applies_prefixes_tags_and_metadata(monkeypatch):
    provider = _make_provider(monkeypatch)

    provider.sync_turn("Need memory labels", "Got it")
    provider._sync_thread.join(timeout=1)

    assert len(provider._client.retain_calls) == 1
    call = provider._client.retain_calls[0]
    assert call["bank_id"] == "josh-global"
    assert call["context"] == "conversation"
    assert call["tags"] == ["agent:abner", "source_system:hermes-agent"]
    assert call["metadata"]["source"] == "hermes"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", call["metadata"]["retained_at"])
    assert call["content"] == "User (Josh): Need memory labels\nAssistant (Abner): Got it"


def test_hindsight_retain_tool_uses_same_tags_and_metadata(monkeypatch):
    provider = _make_provider(monkeypatch)

    result = json.loads(provider.handle_tool_call("hindsight_retain", {"content": "Remember this", "context": "user preference"}))

    assert result == {"result": "Memory stored successfully."}
    assert len(provider._client.retain_calls) == 1
    call = provider._client.retain_calls[0]
    assert call["content"] == "Remember this"
    assert call["context"] == "user preference"
    assert call["tags"] == ["agent:abner", "source_system:hermes-agent"]
    assert call["metadata"]["source"] == "hermes"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", call["metadata"]["retained_at"])
