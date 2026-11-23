"""Tracing hook for the pipeline. Langfuse when configured (LANGFUSE_PUBLIC_KEY / SECRET_KEY /
HOST in .env), otherwise a no-op, so the pipeline never depends on the tracing backend.

Each request is one trace: a root `text2sql` span with child observations for the input guard,
schema building, every LLM call (as a `generation` with token usage), each SQL guard check and
each database execution.
"""

from __future__ import annotations

import contextlib
import logging
import os

log = logging.getLogger("sqlforge.tracing")


class _NoopObs:
    id = None
    trace_id = None

    def update(self, **_):
        return self


class NoopTracer:
    enabled = False

    def attributes(self, **kw):
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def observe(self, name: str, as_type: str = "span", **kw):
        yield _NoopObs()

    def set_trace_io(self, obs, input=None, output=None) -> None:
        pass

    def score(self, trace_id: str, name: str, value, comment: str | None = None) -> bool:
        return False

    def trace_url(self, trace_id: str | None) -> str | None:
        return None

    def flush(self) -> None:
        pass


class LangfuseTracer:
    enabled = True

    def __init__(self):
        from langfuse import Langfuse

        self.lf = Langfuse()

    def attributes(self, **kw):
        """Trace-level name / tags / metadata, propagated to every observation in the trace."""
        from langfuse import propagate_attributes

        return propagate_attributes(**kw)

    @contextlib.contextmanager
    def observe(self, name: str, as_type: str = "span", **kw):
        with self.lf.start_as_current_observation(name=name, as_type=as_type, **kw) as obs:
            yield obs

    def set_trace_io(self, obs, input=None, output=None) -> None:
        """Trace-level input/output, shown in the trace list and header. SDK 5.x has no setter for
        them, but the server still reads these OTel attributes from the root span."""
        import json

        try:
            if input is not None:
                obs._otel_span.set_attribute("langfuse.trace.input", json.dumps(input, default=str))
            if output is not None:
                obs._otel_span.set_attribute("langfuse.trace.output", json.dumps(output, default=str))
        except Exception as e:  # tracing must never break a request
            log.debug("could not set trace input/output: %s", e)

    def score(self, trace_id: str, name: str, value, comment: str | None = None) -> bool:
        # Public REST API instead of the SDK's create_score: SDK 5.x sends a payload the
        # self-hosted v3 server rejects ("Bad request"); the documented endpoint accepts it.
        import base64
        import json
        import urllib.request

        auth = base64.b64encode(f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()).decode()
        body = {"traceId": trace_id, "name": name, "value": float(value), "dataType": "NUMERIC"}
        if comment:
            body["comment"] = comment
        req = urllib.request.Request(os.environ.get("LANGFUSE_HOST", "http://127.0.0.1:3000").rstrip("/") + "/api/public/scores",
                                     data=json.dumps(body).encode(),
                                     headers={"Authorization": "Basic " + auth, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status == 200
        except Exception as e:
            log.warning("could not store score: %s", e)
            return False

    def trace_url(self, trace_id: str | None) -> str | None:
        """Browser link to the trace. In Docker the SDK talks to http://langfuse-web:3000, which a
        browser can't open, so LANGFUSE_PUBLIC_URL (e.g. http://localhost:3000) replaces the host."""
        try:
            url = self.lf.get_trace_url(trace_id=trace_id) if trace_id else None
        except Exception:
            return None
        public, internal = os.environ.get("LANGFUSE_PUBLIC_URL"), os.environ.get("LANGFUSE_HOST")
        if url and public and internal and url.startswith(internal.rstrip("/")):
            url = public.rstrip("/") + url[len(internal.rstrip("/")):]
        return url

    def flush(self) -> None:
        self.lf.flush()


def make_tracer():
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            t = LangfuseTracer()
            log.info("Langfuse tracing enabled (%s)", os.environ.get("LANGFUSE_HOST"))
            return t
        except Exception as e:
            log.warning("Langfuse unavailable, tracing disabled: %s", e)
    return NoopTracer()
