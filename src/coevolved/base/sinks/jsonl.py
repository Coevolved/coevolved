import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TextIO

from coevolved.base.tracing import BaseEvent


def make_verbose_jsonl_serializer(
    *,
    include_none: bool = True,
    sort_keys: bool = True,
    add_event_type: bool = True,
    add_timestamp_iso: bool = True,
    redactor: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
) -> Callable[[BaseEvent], Optional[str]]:
    """Build a verbose JSONL serializer for Coevolved trace events.

    The returned serializer converts a trace event into a single-line JSON string
    with a stable key order (when sort_keys=True). This is a good default for
    JSONL sinks and log ingestion pipelines.

    Args:
        include_none: If True, keep fields whose values are None (more verbose).
            If False, drop None-valued fields.
        sort_keys: If True, sort keys for stable diffs/ingestion.
        add_event_type: If True, add event_type=event.__class__.__name__.
        add_timestamp_iso: If True, add timestamp_iso derived from timestamp.
        redactor: Optional function that can redact or transform the payload dict
            before serialization (e.g., remove secrets).

    Returns:
        A serializer function compatible with JSONLSink(serializer=...).
    """

    def _default_json_serializer(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    def serializer(event: BaseEvent) -> Optional[str]:
        payload: dict[str, Any] = asdict(event)
        if not include_none:
            payload = {k: v for k, v in payload.items() if v is not None}
        if add_event_type:
            payload["event_type"] = event.__class__.__name__
        if add_timestamp_iso and payload.get("timestamp") is not None:
            try:
                payload["timestamp_iso"] = datetime.fromtimestamp(
                    float(payload["timestamp"]), tz=timezone.utc
                ).isoformat()
            except Exception:
                # If timestamp isn't coercible, skip derived field.
                pass
        if redactor:
            payload = redactor(payload)
        return json.dumps(payload, sort_keys=sort_keys, default=_default_json_serializer)

    return serializer


class JSONLSink:
    """JSONL sink that writes one JSON object per line.

    This sink is intentionally minimal: behavior and verbosity are fully controlled
    by the caller-provided serializer.

    Args:
        path: Output file path.
        serializer: Function that converts an event into a single-line JSON string
            (without a trailing newline). Return None/empty to skip writing.
    """

    def __init__(
        self,
        path: str,
        *,
        serializer: Callable[[BaseEvent], Optional[str]],
    ) -> None:
        self._path = path
        self._serializer = serializer
        self._lock = threading.Lock()
        # Open-once; keep the file handle for the sink lifetime.
        self._fp: TextIO = open(self._path, "a", encoding="utf-8")
        self._closed = False

    def emit(self, event: BaseEvent) -> None:
        # Be defensive: sinks should not break application logic.
        try:
            line = self._serializer(event)
        except Exception:
            return
        if not line:
            return
        # Ensure exactly one event per line.
        line = line.rstrip("\n")
        with self._lock:
            if self._closed:
                return
            try:
                self._fp.write(line)
                self._fp.write("\n")
                self._fp.flush()
            except Exception:
                return

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._fp.close()
            except Exception:
                pass

    def __enter__(self) -> "JSONLSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup; never raise.
        try:
            self.close()
        except Exception:
            pass

