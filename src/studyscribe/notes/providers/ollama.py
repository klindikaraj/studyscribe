"""Local models through Ollama's native API (`POST /api/chat`), with standard-library HTTP.

Why the native API and not Ollama's OpenAI-compatible one: only the native API sets the context
window per request (`options.num_ctx`; Ollama's default is 4096 tokens), unloads a model on
request (`keep_alive: 0`) and reports load / prompt / generation times.

Behaviour checked against Ollama 0.40.15 on 2026-11-09 (docs/plan-notes.md, piece 1b):

- A prompt longer than `num_ctx` is cut silently (HTTP 200, the start of the prompt dropped)
  unless the request carries `"truncate": false` at the top level; then Ollama answers 400
  "request (N tokens) exceeds the available context size (M tokens)". Every request here sends
  it, and the provider also refuses to send a request whose estimate does not fit, and discards
  a response whose token counts show a full context.
- `prompt_eval_count` counts every prompt token, cached ones included.
- The last streamed chunk carries the counts and durations; `done_reason` is "stop", "length",
  "load" (a request with no messages loads the model) or "unload" (`keep_alive: 0`).
- A model that does not fit in GPU memory is split between GPU and CPU without an error: after
  loading, `/api/ps` shows `size_vram < size`, and the provider stops unless allowed.
- `num_ctx` below 2048 is raised to 2048 by Ollama.

The provider loads the model once (`ensure_loaded`), keeps it loaded with `keep_alive` while
requests follow each other, and `close()` unloads it at once, so the GPU memory is free again.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import shutil
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from studyscribe.notes import gpu
from studyscribe.notes.providers.base import (
    BadRequest,
    ContextTooLong,
    Effort,
    LlmError,
    LlmRequest,
    LlmResponse,
    NetworkError,
    ServerError,
    Timings,
    Truncated,
    Usage,
)

DEFAULT_PORT = 11434
DEFAULT_HOST = f"http://127.0.0.1:{DEFAULT_PORT}"
DEFAULT_NUM_CTX = 32768
MIN_NUM_CTX = 2048  # Ollama raises smaller values to this
DEFAULT_KEEP_ALIVE = "10m"
DEFAULT_SEED = 1
SHORT_TIMEOUT_S = 15.0  # version, show, tags, ps, unload
LOAD_TIMEOUT_S = 600.0  # first load from disk took 43 s on the RTX 3060
STALL_TIMEOUT_S = 300.0  # longest wait for the next streamed chunk (covers a long prefill)
MARGIN_TOKENS = 512  # kept free in the context besides the output cap
# Characters per token used to estimate a prompt before sending it: deliberately low, so the
# estimate overshoots. Measured 3.4-3.5 with Gemini's tokenizer; the probe measures qwen3.5.
CHARS_PER_TOKEN_MIN = 2.5
# Fewer input tokens than characters / this means the prompt cannot have been read whole.
CHARS_PER_TOKEN_IMPLAUSIBLE = 8.0
DOWNLOAD_PAGE = "https://ollama.com/download/windows"
SMALLER_MODEL = "qwen3.5:4b"

# Sampling per model family (`details.family` of /api/show) and thinking on/off, from the model
# cards. Qwen's cards warn against greedy decoding (endless repetitions), hence no temperature 0;
# the fixed seed makes runs repeatable instead. Unknown families keep the model's own defaults.
SAMPLING: dict[str, dict[bool, dict[str, float]]] = {
    "qwen35": {
        False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
        True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
    },
    # Family name assumed (not checked: the model is not pulled yet).
    "gemma4": {
        False: {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        True: {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    },
}


# --- errors --------------------------------------------------------------------------------------


class OllamaUnavailable(LlmError):
    """Ollama is not running, not installed, or not reachable at the configured host."""


class ModelNotDownloaded(LlmError):
    pass


class OllamaTooOld(LlmError):
    pass


class GpuMemoryShort(LlmError):
    """The model does not fit (entirely) in GPU memory."""


def gpu_remedy(held_by_ollama: Sequence[str] = ()) -> str:
    """What to do when GPU memory is short. Never a smaller context: the notes refuse less
    than 32,768 tokens (2026-11-16, user's manual test)."""
    text = ("Close apps that use the GPU (games, video editors, browsers with hardware "
            f"acceleration), or use a smaller model (--model {SMALLER_MODEL}); --allow-cpu runs "
            "it anyway, slowly.")
    if held_by_ollama:
        names = ", ".join(held_by_ollama)
        text += (f" Ollama still holds {names}: wait until it unloads (up to "
                 f"{DEFAULT_KEEP_ALIVE} after its last use) or run `ollama stop "
                 f"{held_by_ollama[0]}`.")
    return text


# --- configuration -------------------------------------------------------------------------------


def parse_host(value: str | None) -> str:
    """Base URL from an OLLAMA_HOST-style value: "", "0.0.0.0", "127.0.0.1:11434",
    "http://host:port"... A bind-all address means this machine."""
    value = (value or "").strip()
    if not value:
        return DEFAULT_HOST
    if "://" not in value:
        value = "http://" + value
    parts = urlsplit(value)
    scheme = parts.scheme if parts.scheme in ("http", "https") else "http"
    host = parts.hostname or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        port = 443 if scheme == "https" else DEFAULT_PORT
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


def host_from_env(environ: Mapping[str, str] = os.environ) -> str:
    return parse_host(environ.get("OLLAMA_HOST"))


def is_local(base_url: str) -> bool:
    host = (urlsplit(base_url).hostname or "").lower()
    return host in ("localhost", "::1") or host.startswith("127.")


@dataclass(frozen=True)
class OllamaOptions:
    host: str = field(default_factory=host_from_env)
    num_ctx: int = DEFAULT_NUM_CTX
    keep_alive: str = DEFAULT_KEEP_ALIVE
    seed: int = DEFAULT_SEED
    allow_cpu: bool = False
    chars_per_token_min: float = CHARS_PER_TOKEN_MIN
    stall_timeout_s: float = STALL_TIMEOUT_S
    load_timeout_s: float = LOAD_TIMEOUT_S


# --- model information (pure) --------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    name: str
    family: str | None
    parameter_size: str | None
    quantization: str | None
    capabilities: tuple[str, ...]
    thinking_values: tuple[Any, ...] | None  # e.g. (False, True); None when not reported
    context_length: int | None  # the model's maximum
    kv: KvCache | None  # f16 KV cache, when the architecture fields allow it

    @property
    def can_think(self) -> bool:
        return "thinking" in self.capabilities

    def describe(self) -> str:
        bits = [b for b in (self.parameter_size, self.quantization) if b]
        return f"{self.name} ({' '.join(bits)})" if bits else self.name


def model_info_from_show(name: str, body: Mapping[str, Any]) -> ModelInfo:
    details = body.get("details") or {}
    info = body.get("model_info") or {}
    arch = info.get("general.architecture")
    thinking = body.get("thinking")
    values = None
    if isinstance(thinking, Mapping) and isinstance(thinking.get("values"), list):
        values = tuple(thinking["values"])
    context = info.get(f"{arch}.context_length") if arch else None
    return ModelInfo(
        name=name,
        family=details.get("family"),
        parameter_size=details.get("parameter_size"),
        quantization=details.get("quantization_level"),
        capabilities=tuple(body.get("capabilities") or ()),
        thinking_values=values,
        context_length=int(context) if isinstance(context, (int, float)) else None,
        kv=kv_cache(info),
    )


@dataclass(frozen=True)
class KvCache:
    """f16 KV cache of a model: bytes per token of context for the layers that keep every
    token, plus a fixed amount for sliding-window layers (they keep only `window` tokens)."""

    per_token: int
    window_bytes: int = 0

    def bytes_for(self, num_ctx: int) -> int:
        return self.per_token * num_ctx + self.window_bytes


def kv_cache(info: Mapping[str, Any]) -> KvCache | None:
    """KV cache size from /api/show's model_info. Handles per-layer KV head counts (lists),
    sliding-window layers (gemma4: `attention.sliding_window_pattern`, `*_swa` lengths) and
    hybrid models whose other layers keep no KV cache (qwen35: one attention layer every
    `full_attention_interval`)."""
    arch = info.get("general.architecture")
    if not arch:
        return None

    def get(key: str) -> Any:
        return info.get(f"{arch}.{key}")

    def num(key: str) -> float | None:
        value = get(key)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    layers = num("block_count")
    heads = num("attention.head_count")
    embedding = num("embedding_length")
    head_dim = embedding / heads if embedding and heads else None
    key_len = num("attention.key_length") or head_dim
    value_len = num("attention.value_length") or head_dim
    kv_heads = get("attention.head_count_kv")
    if kv_heads is None:
        kv_heads = heads
    if not (layers and key_len and value_len and kv_heads):
        return None
    layers = int(layers)
    pattern = get("attention.sliding_window_pattern")
    window = num("attention.sliding_window")
    key_swa = num("attention.key_length_swa") or key_len
    value_swa = num("attention.value_length_swa") or value_len
    interval = num("full_attention_interval")
    per_token = 0.0
    window_bytes = 0.0
    for layer in range(layers):
        if interval and interval > 1 and (layer + 1) % int(interval) != 0:
            continue  # recurrent layer: a fixed state, no KV cache
        layer_heads = kv_heads[layer] if isinstance(kv_heads, list) else kv_heads
        if not isinstance(layer_heads, (int, float)) or not layer_heads:
            continue
        sliding = (isinstance(pattern, list) and layer < len(pattern) and bool(pattern[layer])
                   and window)
        if sliding:
            window_bytes += layer_heads * (key_swa + value_swa) * 2 * window
        else:
            per_token += layer_heads * (key_len + value_len) * 2
    return KvCache(int(per_token), int(window_bytes))


def think_value(effort: Effort, info: ModelInfo) -> bool | str | None:
    """The `think` field for an effort: None leaves it out (model without thinking)."""
    if not info.can_think:
        if effort == "none":
            return None
        raise BadRequest(f"{info.name} has no thinking mode (effort {effort}).",
                         remedy="Use --effort none with this model.")
    if effort == "none":
        return False
    if info.thinking_values and effort in info.thinking_values:
        return effort  # models with levels ("low", "medium", "high")
    return True


def sampling_for(family: str | None, thinking: bool) -> dict[str, float]:
    return dict(SAMPLING.get(family or "", {}).get(thinking, {}))


# --- errors from HTTP (pure) ---------------------------------------------------------------------

_CONTEXT_RE = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+)")


def error_text(body: bytes | str) -> str:
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("error"), str):
            return data["error"]
    except ValueError:
        pass
    return text.strip()[:500]


def map_error(status: int | None, message: str, model: str, num_ctx: int) -> LlmError:
    low = message.lower()
    match = _CONTEXT_RE.search(message)
    if match:
        tokens = int(match.group(1))
        return ContextTooLong(
            f"The request has {tokens:,} tokens, more than the {int(match.group(2)):,}-token "
            "context: Ollama refused it (nothing was cut).", tokens=tokens, status=status,
            remedy="This is a bug in how the notes are split; please report it.")
    if "not found" in low and ("model" in low or status == 404):
        return ModelNotDownloaded(f"Model {model} is not downloaded.",
                                  remedy=f"Run: ollama pull {model}", status=status)
    if (status == 412 or "newer version" in low or "unknown model architecture" in low
            or "unsupported model" in low):
        return OllamaTooOld(f"This Ollama cannot run {model}: {message}",
                            remedy=f"Update Ollama ({DOWNLOAD_PAGE}).", status=status)
    if any(s in low for s in ("out of memory", "cudamalloc", "insufficient memory",
                              "requires more system memory", "unable to allocate")):
        return GpuMemoryShort(f"Not enough memory to load {model}: {message}",
                              remedy=gpu_remedy(), status=status)
    text = f"Ollama error {status}: {message}" if status else f"Ollama error: {message}"
    if status is None or status >= 500:
        return ServerError(text, status=status,
                           remedy="Ollama's log: %LOCALAPPDATA%\\Ollama\\server.log")
    return BadRequest(text, status=status)


# --- the provider --------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadInfo:
    size_bytes: int  # what Ollama allocated for the model (weights, cache, buffers)
    vram_bytes: int  # the part of it in GPU memory
    context_length: int
    digest: str | None
    load_s: float
    gpu_before: gpu.GpuMemory | None
    gpu_after: gpu.GpuMemory | None

    @property
    def on_gpu_entirely(self) -> bool:
        return self.vram_bytes >= self.size_bytes


@dataclass
class _Stream:
    content: list[str] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)


def find_ollama_exe() -> Path | None:
    found = shutil.which("ollama")
    if found:
        return Path(found)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = Path(local) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            return candidate
    return None


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, options: OllamaOptions | None = None, *,
                 on_progress: Callable[[int, float], None] | None = None,
                 find_exe: Callable[[], Path | None] = find_ollama_exe,
                 gpu_query: Callable[[], gpu.GpuMemory | None] = gpu.query,
                 on_notice: Callable[[str], None] | None = None) -> None:
        self.model = model
        self.options = options or OllamaOptions()
        self.on_progress = on_progress  # (chunks received, seconds) while an answer streams
        # Told about what loading does with models Ollama already holds; kept in `notices`.
        self.on_notice = on_notice
        self.notices: list[str] = []
        self._find_exe = find_exe
        self._gpu_query = gpu_query
        url = urlsplit(self.options.host)
        self._https = url.scheme == "https"
        self._host = url.hostname or "127.0.0.1"
        self._port = url.port or DEFAULT_PORT
        self.version: str | None = None
        self.info: ModelInfo | None = None
        self.load_info: LoadInfo | None = None
        self._load_attempted = False

    # --- LlmProvider -----------------------------------------------------------------------------

    def check(self) -> str:
        version = self.server_version()
        return f"{self.model_info().describe()}, Ollama {version}"

    def generate(self, request: LlmRequest) -> LlmResponse:
        info = self.model_info()
        think = think_value(request.effort, info)
        self.check_fits(request.system, request.user, request.max_output_tokens,
                        counted=request.input_tokens)
        self.ensure_loaded()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": _messages(request.system, request.user),
            "format": request.schema,
            "options": {
                **sampling_for(info.family, bool(think)),
                "num_ctx": self.options.num_ctx,
                "num_predict": request.max_output_tokens,
                "seed": self.options.seed,
            },
        }
        if think is not None:
            body["think"] = think
        started = time.perf_counter()
        stream = self._chat(body)
        duration = time.perf_counter() - started
        final = stream.final
        # Ollama's eval_count covers thinking and answer together. A streamed chunk carries about
        # one token (29 content chunks for an eval_count of 33 in a check on 0.40.15: special
        # tokens are not streamed), so the thinking chunks approximate the thinking tokens.
        # None with thinking off.
        usage = Usage(input_tokens=int(final.get("prompt_eval_count") or 0),
                      output_tokens=int(final.get("eval_count") or 0),
                      thinking_tokens=len(stream.thinking) if think else None,
                      cached_tokens=int(final.get("prompt_eval_cached_count") or 0))
        timings = Timings(load_s=_seconds(final.get("load_duration")),
                          prefill_s=_seconds(final.get("prompt_eval_duration")),
                          generate_s=_seconds(final.get("eval_duration")))
        if final.get("done_reason") == "length":
            still_thinking = bool(stream.thinking) and not "".join(stream.content).strip()
            where = (f", while still thinking ({len(stream.thinking):,} thinking tokens)"
                     if still_thinking else "")
            raise Truncated(f"The answer was cut at {request.max_output_tokens:,} output "
                            f"tokens{where}.", usage=usage,
                            remedy="Thinking needs a larger output cap, or use --effort none."
                            if still_thinking else "The notes pipeline asks again with more room.")
        self._check_not_cut(usage, len(request.system) + len(request.user))
        text = "".join(stream.content)
        if not text.strip():
            raise LlmError(f"{self.model} returned an empty answer.", usage=usage,
                           remedy="Try again; if it persists, try another model.")
        digest = self.load_info.digest if self.load_info else None
        return LlmResponse(text=text, usage=usage,
                           model_version=f"{self.model}@{digest[:12]}" if digest else self.model,
                           request_id=None, duration_s=duration, timings=timings)

    def count_tokens(self, system: str, user: str) -> int:
        """Exact input tokens (chat template included): Ollama's own count, from a one-token
        answer, or from its refusal when the text is longer than the context."""
        self.ensure_loaded()
        body = {"model": self.model, "messages": _messages(system, user),
                "options": {"num_ctx": self.options.num_ctx, "num_predict": 1,
                            "seed": self.options.seed}}
        if self.model_info().can_think:
            body["think"] = False
        try:
            return int(self._chat(body).final.get("prompt_eval_count") or 0)
        except ContextTooLong as error:
            if error.tokens is not None:
                return error.tokens
            raise

    def close(self) -> None:
        if self._load_attempted:
            try:
                self.unload()
            except Exception:  # never mask the error that led here
                pass

    # --- steps -----------------------------------------------------------------------------------

    def server_version(self) -> str:
        if self.version is None:
            data = self._json("GET", "/api/version")
            self.version = str(data.get("version", "unknown"))
        return self.version

    def model_info(self) -> ModelInfo:
        if self.info is None:
            data = self._json("POST", "/api/show", {"model": self.model})
            self.info = model_info_from_show(self.model, data)
        return self.info

    def downloaded_models(self) -> list[tuple[str, int]]:
        data = self._json("GET", "/api/tags")
        return [(m.get("name") or m.get("model") or "?", int(m.get("size") or 0))
                for m in data.get("models") or []]

    def running_models(self) -> list[dict[str, Any]]:
        return list(self._json("GET", "/api/ps").get("models") or [])

    def budget(self, max_output_tokens: int) -> int:
        """Input tokens a request may have with this output cap."""
        return self.options.num_ctx - max_output_tokens - MARGIN_TOKENS

    def estimate_tokens(self, system: str, user: str) -> int:
        return math.ceil((len(system) + len(user)) / self.options.chars_per_token_min) + 16

    def check_fits(self, system: str, user: str, max_output_tokens: int, *,
                   counted: int | None = None) -> None:
        """Guard 1: refuse before sending. `counted`: the exact input tokens, when the caller
        counted them; otherwise a conservative estimate from the characters."""
        estimate = counted if counted is not None else self.estimate_tokens(system, user)
        budget = self.budget(max_output_tokens)
        if estimate > budget:
            how = "" if counted is not None else "about "
            raise ContextTooLong(
                f"The request ({how}{estimate:,} tokens) does not fit the context: "
                f"{self.options.num_ctx:,} tokens minus {max_output_tokens:,} for the answer. "
                "Nothing was sent.",
                remedy="This is a bug in how the notes are split; please report it.")

    def vram_needed_bytes(self) -> int | None:
        """Rough need before loading: download size plus the KV cache for num_ctx."""
        info = self.model_info()
        sizes = dict(self.downloaded_models())
        size = sizes.get(self.model) or sizes.get(_with_tag(self.model))
        if not size:
            return None
        return size + (info.kv.bytes_for(self.options.num_ctx) if info.kv else 0)

    def ensure_loaded(self) -> LoadInfo:
        """Load the model with the run's context window and check it sits entirely in GPU
        memory. Raises GpuMemoryShort (after unloading) unless allow_cpu."""
        if self.load_info is not None:
            return self.load_info
        self.server_version()
        self.model_info()
        before = self._gpu_query()
        need = self.vram_needed_bytes()
        held = self._held_by_ollama()
        if held.ready:
            self._notice(f"{self.model} is already loaded in Ollama with a "
                         f"{self.options.num_ctx:,}-token context: reused.")
        else:
            for line in held.describe(self.model):
                self._notice(line)
            free = None if before is None else before.free_mib * gpu.MIB + held.reclaimable
            if (free is not None and need is not None and not self.options.allow_cpu
                    and free < 0.75 * need):
                ollama = (f", {held.reclaimable / 1e9:.1f} GB of them held by Ollama and "
                          "counted as free" if held.reclaimable else "")
                raise GpuMemoryShort(
                    f"{self.model} needs about {need / 1e9:.1f} GB of GPU memory with a "
                    f"{self.options.num_ctx:,}-token context; {free / 1e9:.1f} GB can be free "
                    f"({before.used_mib / 1024:.1f} of {before.total_mib / 1024:.1f} GB in "
                    f"use{ollama}).", remedy=gpu_remedy())
        self._load_attempted = True
        started = time.perf_counter()
        self._json("POST", "/api/chat", {"model": self.model, "messages": [],
                                         "keep_alive": self.options.keep_alive,
                                         "options": {"num_ctx": self.options.num_ctx}},
                   timeout=self.options.load_timeout_s)
        load_s = time.perf_counter() - started
        entry = _find_running(self.running_models(), self.model)
        if entry is None:
            raise LlmError(f"Ollama does not list {self.model} as loaded after loading it.",
                           remedy="Ollama's log: %LOCALAPPDATA%\\Ollama\\server.log")
        self.load_info = LoadInfo(
            size_bytes=int(entry.get("size") or 0), vram_bytes=int(entry.get("size_vram") or 0),
            context_length=int(entry.get("context_length") or 0), digest=entry.get("digest"),
            load_s=load_s, gpu_before=before, gpu_after=self._gpu_query())
        info = self.load_info
        if not info.on_gpu_entirely and not self.options.allow_cpu:
            others = [name for name in self._held_by_ollama().names if
                      _with_tag(name) != _with_tag(self.model)]
            self.close()
            raise GpuMemoryShort(
                f"{self.model} fits only partly in GPU memory ({info.vram_bytes / 1e9:.1f} of "
                f"{info.size_bytes / 1e9:.1f} GB): the rest would run on the CPU, many times "
                "slower.", remedy=gpu_remedy(others))
        if 0 < info.context_length < self.options.num_ctx:
            self.close()
            raise BadRequest(f"Ollama loaded {self.model} with a {info.context_length:,}-token "
                             f"context instead of {self.options.num_ctx:,}.",
                             remedy="Check OLLAMA_CONTEXT_LENGTH in Ollama's settings: it "
                             f"must allow {self.options.num_ctx:,} tokens (or be unset); then "
                             "restart Ollama.")
        return info

    def _held_by_ollama(self) -> HeldByOllama:
        """What Ollama holds in GPU memory now (`/api/ps`); nothing when it cannot say."""
        try:
            running = self.running_models()
        except LlmError:
            running = []
        return HeldByOllama.from_ps(running, self.model, self.options.num_ctx)

    def _notice(self, text: str) -> None:
        self.notices.append(text)
        if self.on_notice is not None:
            self.on_notice(text)

    def unload(self) -> None:
        self._json("POST", "/api/chat", {"model": self.model, "messages": [], "keep_alive": 0})
        self.load_info = None
        self._load_attempted = False

    # --- HTTP ------------------------------------------------------------------------------------

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=timeout)

    def _json(self, method: str, path: str, body: Any = None,
              timeout: float = SHORT_TIMEOUT_S) -> dict[str, Any]:
        conn = self._connection(timeout)
        try:
            response = self._send(conn, method, path, body)
            data = response.read()
        except (TimeoutError, socket.timeout):
            raise NetworkError(f"Ollama did not answer within {timeout:.0f} s ({path}).",
                               remedy="Check that Ollama is not stuck (restart it).") from None
        except OSError as error:
            raise self._connection_error(error) from None
        finally:
            conn.close()
        if response.status >= 400:
            raise map_error(response.status, error_text(data), self.model, self.options.num_ctx)
        try:
            parsed = json.loads(data or b"{}")
        except ValueError:
            raise ServerError(f"Ollama sent something that is not JSON ({path}).") from None
        return parsed if isinstance(parsed, dict) else {}

    def _chat(self, body: dict[str, Any]) -> _Stream:
        """Streamed /api/chat: every request refuses truncation and keeps the model loaded."""
        body = {**body, "stream": True, "truncate": False, "keep_alive": self.options.keep_alive}
        conn = self._connection(self.options.stall_timeout_s)
        stream = _Stream()
        started = time.perf_counter()
        chunks = 0
        try:
            response = self._send(conn, "POST", "/api/chat", body)
            if response.status >= 400:
                raise map_error(response.status, error_text(response.read()), self.model,
                                self.options.num_ctx)
            while True:
                line = response.readline()
                if not line:
                    raise NetworkError("Ollama closed the answer before the end.",
                                       remedy="Ollama may have crashed: see "
                                       "%LOCALAPPDATA%\\Ollama\\server.log")
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    raise ServerError("Ollama sent a malformed chunk.") from None
                if isinstance(chunk.get("error"), str):
                    raise map_error(None, chunk["error"], self.model, self.options.num_ctx)
                message = chunk.get("message") or {}
                if message.get("content"):
                    stream.content.append(message["content"])
                if message.get("thinking"):
                    stream.thinking.append(message["thinking"])
                chunks += 1
                if self.on_progress is not None:
                    self.on_progress(chunks, time.perf_counter() - started)
                if chunk.get("done"):
                    stream.final = chunk
                    return stream
        except (TimeoutError, socket.timeout):
            raise NetworkError(
                f"Ollama sent nothing for {self.options.stall_timeout_s:.0f} s.",
                remedy="Check that Ollama is not stuck (restart it).") from None
        except OSError as error:
            raise self._connection_error(error) from None
        finally:
            conn.close()

    @staticmethod
    def _send(conn: http.client.HTTPConnection, method: str, path: str,
              body: Any) -> http.client.HTTPResponse:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        conn.request(method, path, body=payload, headers=headers)
        return conn.getresponse()

    def _connection_error(self, error: OSError) -> LlmError:
        where = self.options.host
        if isinstance(error, ConnectionRefusedError):
            if not is_local(where):
                return OllamaUnavailable(f"Ollama is not reachable at {where}.",
                                         remedy="Check OLLAMA_HOST and that Ollama runs there.")
            if self._find_exe() is None:
                return OllamaUnavailable("Ollama is not installed.",
                                         remedy=f"Install it from {DOWNLOAD_PAGE}")
            return OllamaUnavailable(f"Ollama is not running ({where}).",
                                     remedy="Start Ollama from the Start menu, or run "
                                     "`ollama serve`.")
        return NetworkError(f"Connection to Ollama lost: {type(error).__name__}: {error}",
                            remedy="Ollama may have crashed or restarted; try again.")

    def _check_not_cut(self, usage: Usage, chars: int) -> None:
        if usage.input_tokens + usage.output_tokens >= self.options.num_ctx:
            raise ContextTooLong(
                f"The context was full ({usage.input_tokens:,} input + {usage.output_tokens:,} "
                f"output tokens of {self.options.num_ctx:,}): the input may have been cut, so the "
                "answer is discarded.", tokens=usage.input_tokens, usage=usage,
                remedy="This is a bug in how the notes are split; please report it.")
        if chars and usage.input_tokens < chars / CHARS_PER_TOKEN_IMPLAUSIBLE:
            raise ContextTooLong(
                f"Ollama read only {usage.input_tokens:,} tokens of a {chars:,}-character prompt: "
                "the input was probably cut, so the answer is discarded.",
                tokens=usage.input_tokens, usage=usage,
                remedy="Update Ollama; this version may ignore `truncate: false`.")


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _seconds(nanoseconds: Any) -> float:
    return float(nanoseconds) / 1e9 if isinstance(nanoseconds, (int, float)) else 0.0


def _with_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


@dataclass(frozen=True)
class HeldByOllama:
    """The models Ollama keeps loaded, as `/api/ps` lists them. Their GPU memory is
    reclaimable: Ollama unloads a model to load another one, or the same one with another
    context. `ready`: our model is there already, with our context, entirely in GPU memory
    (after a killed run it stays loaded for keep_alive): nothing to free, nothing to check."""

    ready: bool
    reclaimable: int  # bytes of GPU memory held by the models that would be unloaded
    entries: tuple[tuple[str, int, int], ...]  # (name, context length, bytes in GPU memory)

    @property
    def names(self) -> list[str]:
        return [name for name, _, _ in self.entries]

    @classmethod
    def from_ps(cls, running: list[dict[str, Any]], model: str, num_ctx: int) -> HeldByOllama:
        entries = tuple((str(e.get("name") or e.get("model") or "?"),
                         int(e.get("context_length") or 0), int(e.get("size_vram") or 0))
                        for e in running)
        same = _find_running(running, model)
        ready = (same is not None and int(same.get("context_length") or 0) == num_ctx
                 and int(same.get("size_vram") or 0) >= int(same.get("size") or 0) > 0)
        return cls(ready, 0 if ready else sum(vram for _, _, vram in entries), entries)

    def describe(self, model: str) -> list[str]:
        lines = []
        for name, context, vram in self.entries:
            if _with_tag(name) == _with_tag(model):
                why = f"with a {context:,}-token context: Ollama reloads it with the new one"
            else:
                why = "another model: Ollama unloads it to make room"
            lines.append(f"Ollama holds {name} ({vram / 1e9:.1f} GB of GPU memory), {why}; "
                         "that memory is counted as free.")
        return lines


def _find_running(models: list[dict[str, Any]], model: str) -> dict[str, Any] | None:
    wanted = _with_tag(model)
    for entry in models:
        if _with_tag(str(entry.get("model") or entry.get("name") or "")) == wanted:
            return entry
    return None
