"""Compression dumps — the human/AI review workflow's entry point.

Reports answer "which number looks suspicious"; dumps answer "was the
compression actually any good" by writing the *full text* of every
compression event to disk: the message list before, the message list
after, and the summary that replaced them.

The intended reviewer is deliberately not configured as a model (yet):
the files are plain Markdown on local disk so a human — or an AI assistant
reading the workspace — can open them, compare before/after, and issue
verdicts the numeric pipeline cannot (subtle omissions, misattributed
causes, tone loss in Chinese paraphrase, ...). When a judge model is
eventually wired in, these same files are the grounding it would read.

Files land under ``<dump_dir>/<run_id>/<scenario_id>__<arm>.md``; the
data source is the same ``post_compress_hook`` telemetry every other
metric uses — dumping is a side channel off the recorder, zero extra
intrusion into the package.
"""
from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Self

from langchain_core.messages import BaseMessage

from benchmarks.probes import render_context
from benchmarks.telemetry import CompressEvent

# Four-backtick fences: message bodies and LLM summaries may themselves
# contain triple-backtick code blocks, which must not terminate the fence.
_FENCE_OPEN = "````text"
_FENCE_CLOSE = "````"

_KIND_NOTE = {
    "l3": "clean L3 summarization",
    "l0_only": "L0 rule-based cleanup only (no summary)",
    "degraded": "degraded fallback — see plan",
}


def _label(message: BaseMessage) -> str:
    """Render tag for one message: type, identity, and any bench-visible marks."""
    tag = str(getattr(message, "type", "message"))
    ak = getattr(message, "additional_kwargs", None) or {}
    if isinstance(ak, dict):
        if ak.get("__summarization__") or ak.get("lc_source") == "summarization":
            tag = "SUMMARY"
        if isinstance(ak.get("degradation"), dict):
            tag += " [degraded]"
    ident = []
    mid = getattr(message, "id", None)
    if mid:
        ident.append(f"id={mid}")
    tcid = getattr(message, "tool_call_id", None)
    if tcid:
        ident.append(f"tool_call_id={tcid}")
    name = getattr(message, "name", None)
    if name:
        ident.append(f"name={name}")
    return f"{tag} ({', '.join(ident)})" if ident else tag


def _body(message: BaseMessage) -> str:
    """Full text of one message — nothing truncated, nothing stripped.

    Unlike the probe-side renderer, reasoning content-parts are kept:
    the dump documents *what existed before compression*, and L0's
    stripping of reasoning is itself one of the things under review.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "reasoning":
                    parts.append(f"[reasoning] {part.get('reasoning') or part.get('text') or ''}")
                else:
                    parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        text = "\n".join(p for p in parts if p)
    else:
        text = str(content or "")
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        rendered = "; ".join(
            f"{c.get('name')}({_json_compact(c.get('args'))})" for c in calls
        )
        text = f"{text}\n[tool_calls] {rendered}" if text else f"[tool_calls] {rendered}"
    return text


def _json_compact(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _write_messages(handle: IO[str], messages: list[BaseMessage]) -> None:
    for i, m in enumerate(messages):
        handle.write(
            f"\n**[{i:03d}] {_label(m)}**\n\n{_FENCE_OPEN}\n{_body(m)}\n{_FENCE_CLOSE}\n"
        )


def _entry_body(entry: Any) -> str:
    """Full text of one transcript entry (mirrors ``_body`` for scenarios)."""
    content = entry.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "reasoning":
                    parts.append(f"[reasoning] {part.get('reasoning') or part.get('text') or ''}")
                else:
                    parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        text = "\n".join(p for p in parts if p)
    else:
        text = str(content or "")
    calls = entry.tool_calls or []
    if calls:
        rendered = "; ".join(
            f"{c.get('name')}({_json_compact(c.get('args'))})" for c in calls
        )
        text = f"{text}\n[tool_calls] {rendered}" if text else f"[tool_calls] {rendered}"
    return text


def _entry_label(entry: Any, index: int) -> str:
    ident = []
    if entry.tool_call_id:
        ident.append(f"tool_call_id={entry.tool_call_id}")
    if entry.name:
        ident.append(f"name={entry.name}")
    tag = f"t{index} {entry.role}"
    return f"{tag} ({', '.join(ident)})" if ident else tag


class DumpWriter:
    """Writes one (scenario × arm) dump file across its whole replay."""

    def __init__(self, path: Path, scenario_id: str, arm: str) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")
        self._handle.write(
            f"# Compression dump — {scenario_id} × `{arm}`\n\n"
            f"Self-contained review material: the original conversation, the "
            f"ground-truth checklist, every compression event's before/after, "
            f"and the final context a downstream answerer would see.\n"
        )

    def write_original(self, scenario: Any) -> None:
        """The complete pre-compression conversation, nothing stripped.

        This is the reference a third-party judge reads against the
        summary/final context: every fact below must trace back to here.
        """
        n = len(scenario.transcript)
        self._handle.write(
            f"\n---\n\n## Original conversation ({n} entries, pre-compression)\n"
        )
        for i, entry in enumerate(scenario.transcript):
            self._handle.write(
                f"\n**[{i:03d}] {_entry_label(entry, i)}**\n\n"
                f"{_FENCE_OPEN}\n{_entry_body(entry)}\n{_FENCE_CLOSE}\n"
            )
        self._handle.flush()

    def write_ground_truth(self, scenario: Any) -> None:
        """Facts and entities the compression is expected to preserve.

        Scoring anchors for a third-party judge: mark each fact
        preserved / partial / lost against the summary text, and each
        entity by whether its *latest* value survives.
        """
        self._handle.write("\n---\n\n## Ground truth — facts to preserve\n\n")
        if scenario.facts:
            for fact in scenario.facts:
                src = f" (msg {fact.message_ids})" if fact.message_ids else ""
                self._handle.write(f"- [ ] `{fact.id}`{src} {fact.text}\n")
        else:
            self._handle.write("- (none)\n")
        self._handle.write("\n## Ground truth — entities (latest value must survive)\n\n")
        if scenario.entities:
            for ent in scenario.entities:
                aliases = f" | aka: {', '.join(ent.aliases)}" if ent.aliases else ""
                self._handle.write(f"- `{ent.name}` → {ent.value}{aliases}\n")
        else:
            self._handle.write("- (none)\n")
        self._handle.flush()

    def write_event(
        self,
        event: CompressEvent,
        before: list[BaseMessage],
        after: list[BaseMessage],
    ) -> None:
        note = _KIND_NOTE.get(event.kind, event.kind)
        self._handle.write(
            f"\n---\n\n## Event #{event.seq} — turn {event.turn}, {note}\n\n"
            f"- tokens: {event.pre_tokens} → {event.post_tokens}"
            f" (L0 cut {event.l0_delta}, L3 cut {event.l3_delta})\n"
            f"- messages: {len(before)} → {len(after)}"
            f" ({len(event.summarized_keys)} summarized away,"
            f" {len(event.preserved_keys)} preserved)\n"
            + (
                f"- plan: {event.plan}"
                f" — {event.degradation_reason}"
                + (f" (external_ref: {event.external_ref})" if event.external_ref else "")
                + "\n"
                if event.plan
                else ""
            )
        )
        self._handle.write("\n### BEFORE (what the agent had)\n")
        _write_messages(self._handle, before)
        self._handle.write("\n### AFTER (what replaced it)\n")
        _write_messages(self._handle, after)
        if event.summary_text:
            self._handle.write(
                f"\n### Summary text\n\n{_FENCE_OPEN}\n{event.summary_text}\n{_FENCE_CLOSE}\n"
            )
        self._handle.flush()

    def write_final(self, final_messages: list[BaseMessage]) -> None:
        """The reconstructed context exactly as the probe answerer sees it."""
        self._handle.write(
            "\n---\n\n## Final context (as the probe answerer sees it)\n\n"
            f"{_FENCE_OPEN}\n{render_context(final_messages)}\n{_FENCE_CLOSE}\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["DumpWriter"]
