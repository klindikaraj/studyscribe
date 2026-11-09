"""Ollama provider against a fake Ollama: a small HTTP server on a local socket that answers
like Ollama 0.40.15 did on 2026-11-09 (streamed NDJSON in chunked encoding, the same error
texts). No Ollama, no GPU, no network."""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from studyscribe.notes import gpu, pricing
from studyscribe.notes.providers import make_provider
from studyscribe.notes.providers.base import (
    BadRequest,
    ContextTooLong,
    LlmRequest,
    MissingKey,
    NetworkError,
    ServerError,
    Truncated,
    Usage,
)
from studyscribe.notes.providers.ollama import (
    GpuMemoryShort,
    ModelNotDownloaded,
    OllamaOptions,
    OllamaProvider,
    OllamaTooOld,
    OllamaUnavailable,
    is_local,
    kv_bytes_per_token,
    map_error,
    model_info_from_show,
    parse_host,
)

MODEL = "qwen3.5:9b"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SCHEMA = {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}
REQUEST = LlmRequest(system="Describe the passage.", user="S1 the loss dot backward call.",
                     schema=SCHEMA, max_output_tokens=1000, effort="none")

QWEN_INFO = {
    "general.architecture": "qwen35", "qwen35.block_count": 32, "qwen35.context_length": 262144,
    "qwen35.embedding_length": 4096, "qwen35.attention.head_count": 16,
    "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256, "qwen35.full_attention_interval": 4,
}
QWEN_SHOW = {
    "capabilities": ["completion", "vision", "tools", "thinking"],
    "thinking": {"values": [False, True], "default": True},
    "details": {"family": "qwen35", "parameter_size": "9.0B", "quantization_level": "Q4_K_M"},
    "model_info": QWEN_INFO,
}
GB = 1_000_000_000


def final_chunk(prompt=31, out=33, reason="stop", cached=0):
    return {"model": MODEL, "message": {"role": "assistant", "content": ""}, "done": True,
            "done_reason": reason, "total_duration": 3_000_000_000, "load_duration": 100_000_000,
            "prompt_eval_count": prompt, "prompt_eval_cached_count": cached,
            "prompt_eval_duration": 200_000_000, "eval_count": out,
            "eval_duration": 1_100_000_000}


def content(text):
    return {"model": MODEL, "message": {"role": "assistant", "content": text}, "done": False}


class FakeOllama:
    """Routes: GET /api/version, /api/tags, /api/ps; POST /api/show, /api/chat. Chat requests
    with no messages load or unload the model; the others pop `self.answers`."""

    def __init__(self):
        self.requests: list[tuple[str, str, dict | None]] = []
        # ("stream", [chunks]) | ("cut", [chunks]) | (status, body) | ("sleep", seconds)
        self.answers: list = []
        self.show: tuple[int, dict] = (200, QWEN_SHOW)
        self.load_answer: tuple[int, dict] | None = None
        self.size = int(5.6 * GB)
        self.size_vram = int(5.6 * GB)
        self.context_length = 32768
        self.loaded = False

    def chats(self):
        return [body for method, path, body in self.requests
                if path == "/api/chat" and body and body.get("messages")]

    def handle(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        body = None
        if method == "POST":
            length = int(handler.headers.get("Content-Length") or 0)
            body = json.loads(handler.rfile.read(length) or b"null")
        path = handler.path
        self.requests.append((method, path, body))
        if path == "/api/version":
            return send_json(handler, 200, {"version": "0.40.15"})
        if path == "/api/tags":
            return send_json(handler, 200, {"models": [{"name": MODEL, "size": int(6.55 * GB)}]})
        if path == "/api/ps":
            models = [{"name": MODEL, "model": MODEL, "size": self.size,
                       "size_vram": self.size_vram, "digest": DIGEST,
                       "context_length": self.context_length}] if self.loaded else []
            return send_json(handler, 200, {"models": models})
        if path == "/api/show":
            return send_json(handler, *self.show)
        if path == "/api/chat" and not body.get("messages"):
            if body.get("keep_alive") == 0:
                self.loaded = False
                return send_json(handler, 200, {"done": True, "done_reason": "unload"})
            if self.load_answer is not None:
                return send_json(handler, *self.load_answer)
            self.loaded = True
            return send_json(handler, 200, {"done": True, "done_reason": "load"})
        if path == "/api/chat":
            kind, payload = self.answers.pop(0)
            if kind == "sleep":
                time.sleep(payload)
                return send_json(handler, 200, {})
            if kind in ("stream", "cut"):
                return send_stream(handler, payload)
            return send_json(handler, kind, payload)
        return send_json(handler, 404, {"error": "404 page not found"})


def send_json(handler, status, body):
    data = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Connection", "close")
    handler.close_connection = True
    handler.end_headers()
    handler.wfile.write(data)


def send_stream(handler, chunks):
    handler.send_response(200)
    handler.send_header("Content-Type", "application/x-ndjson")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.send_header("Connection", "close")
    handler.close_connection = True
    handler.end_headers()
    for chunk in chunks:
        line = json.dumps(chunk).encode() + b"\n"
        handler.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
    handler.wfile.write(b"0\r\n\r\n")


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass  # the client closing first (timeouts, early close) is part of the tests


@pytest.fixture
def fake():
    state = FakeOllama()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            state.handle(self, "GET")

        def do_POST(self):
            state.handle(self, "POST")

        def log_message(self, *args):
            pass

    server = QuietServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    state.host = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def provider_for(fake, *, free_mib=10_000, **options):
    gpu_state = gpu.GpuMemory("RTX 3060", 12288 - free_mib, 12288)
    return OllamaProvider(MODEL, OllamaOptions(host=fake.host, **options),
                          gpu_query=lambda: gpu_state, find_exe=lambda: Path("ollama.exe"))


# --- generate ------------------------------------------------------------------------------------


def test_generate_loads_once_streams_and_reports_usage_and_timings(fake):
    fake.answers = [("stream", [content('{"topic":'), content(' "gradients"}'), final_chunk()]),
                    ("stream", [content('{"topic": "x"}'), final_chunk()])]
    progress = []
    provider = provider_for(fake)
    provider.on_progress = lambda chunks, seconds: progress.append(chunks)

    response = provider.generate(REQUEST)
    provider.generate(REQUEST)

    assert json.loads(response.text) == {"topic": "gradients"}
    assert response.usage == Usage(input_tokens=31, output_tokens=33)
    assert response.timings.prefill_s == pytest.approx(0.2)
    assert response.timings.generate_rate(33) == pytest.approx(30.0)
    assert response.model_version == f"{MODEL}@{DIGEST[:12]}"
    assert progress[:3] == [1, 2, 3]
    loads = [b for m, p, b in fake.requests if p == "/api/chat" and b["messages"] == []]
    assert loads == [{"model": MODEL, "messages": [], "keep_alive": "10m",
                      "options": {"num_ctx": 32768}}]
    body = fake.chats()[0]
    assert body["stream"] is True and body["truncate"] is False and body["think"] is False
    assert body["keep_alive"] == "10m" and body["format"] == SCHEMA
    assert body["options"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                               "presence_penalty": 1.5, "num_ctx": 32768, "num_predict": 1000,
                               "seed": 1}
    assert body["messages"] == [{"role": "system", "content": REQUEST.system},
                                {"role": "user", "content": REQUEST.user}]
    assert provider.load_info.on_gpu_entirely


def test_close_unloads_the_model(fake):
    fake.answers = [("stream", [content('{"topic": "x"}'), final_chunk()])]
    provider = provider_for(fake)
    provider.generate(REQUEST)
    provider.close()
    assert fake.requests[-1] == ("POST", "/api/chat",
                                 {"model": MODEL, "messages": [], "keep_alive": 0})
    assert not fake.loaded and provider.load_info is None


def test_close_without_loading_sends_nothing(fake):
    provider = provider_for(fake)
    provider.check()
    provider.close()
    assert all(path != "/api/chat" for _, path, _ in fake.requests)


def test_thinking_effort_turns_thinking_on_with_its_sampling(fake):
    fake.answers = [("stream", [content('{"topic": "x"}'), final_chunk()])]
    provider_for(fake).generate(LlmRequest("s", "S1 text", SCHEMA, 1000, effort="high"))
    body = fake.chats()[0]
    assert body["think"] is True
    assert body["options"]["temperature"] == 1.0 and body["options"]["top_p"] == 0.95


def thinking(text):
    return {"model": MODEL, "message": {"role": "assistant", "content": "", "thinking": text},
            "done": False}


def test_thinking_tokens_are_counted_from_the_thinking_chunks(fake):
    fake.answers = [("stream", [thinking("The"), thinking(" user"), thinking(" wants"),
                                content('{"topic":'), content(' "x"}'), final_chunk(out=5)]),
                    ("stream", [content('{"topic": "x"}'), final_chunk(out=1)])]
    provider = provider_for(fake)
    response = provider.generate(LlmRequest("s", "S1 text", SCHEMA, 1000, effort="high"))
    assert response.usage.thinking_tokens == 3 and response.usage.output_tokens == 5
    assert json.loads(response.text) == {"topic": "x"}
    assert provider.generate(REQUEST).usage.thinking_tokens is None


def test_cut_while_thinking_says_so(fake):
    fake.answers = [("stream", [thinking("Hmm"), thinking(" more"),
                                final_chunk(out=6144, reason="length")])]
    with pytest.raises(Truncated, match="while still thinking") as info:
        provider_for(fake).generate(LlmRequest("s", "S1 text", SCHEMA, 6144, effort="high"))
    assert "larger output cap" in info.value.remedy


def test_model_without_thinking(fake):
    show = json.loads(json.dumps(QWEN_SHOW))
    show["capabilities"] = ["completion"]
    show.pop("thinking")
    show["details"]["family"] = "llama"
    fake.show = (200, show)
    fake.answers = [("stream", [content('{"topic": "x"}'), final_chunk()])]
    provider = provider_for(fake)
    provider.generate(REQUEST)
    body = fake.chats()[0]
    assert "think" not in body
    assert set(body["options"]) == {"num_ctx", "num_predict", "seed"}  # model's own sampling
    with pytest.raises(BadRequest, match="no thinking mode"):
        provider.generate(LlmRequest("s", "S1 text", SCHEMA, 1000, effort="medium"))


def test_output_cut_at_num_predict_is_truncated(fake):
    fake.answers = [("stream", [content('{"topic": "gra'), final_chunk(out=1000,
                                                                       reason="length")])]
    with pytest.raises(Truncated) as info:
        provider_for(fake).generate(REQUEST)
    assert info.value.usage.output_tokens == 1000


def test_empty_answer_is_an_error(fake):
    fake.answers = [("stream", [final_chunk(out=1)])]
    with pytest.raises(Exception, match="empty answer"):
        provider_for(fake).generate(REQUEST)


# --- context guards ------------------------------------------------------------------------------


def test_request_that_cannot_fit_is_not_sent(fake):
    provider = provider_for(fake, num_ctx=4096)
    long_user = "S1 " + "word " * 4000  # 20k characters: about 8k tokens at 2.5 chars/token
    with pytest.raises(ContextTooLong, match="Nothing was sent"):
        provider.generate(LlmRequest("sys", long_user, SCHEMA, 1000))
    assert fake.chats() == [] and not fake.loaded


def test_server_refusal_of_a_long_prompt_is_context_too_long(fake):
    fake.answers = [(400, {"error": "request (40141 tokens) exceeds the available context "
                                    "size (32768 tokens), try increasing it"})]
    with pytest.raises(ContextTooLong) as info:
        provider_for(fake).generate(REQUEST)
    assert info.value.tokens == 40141 and not info.value.retryable


def test_full_context_discards_the_answer(fake):
    fake.answers = [("stream", [content('{"topic": "x"}'), final_chunk(prompt=31_800, out=968)])]
    with pytest.raises(ContextTooLong, match="context was full"):
        provider_for(fake).generate(REQUEST)


def test_implausibly_few_input_tokens_discard_the_answer(fake):
    user = "S1 " + "closures and scope " * 200  # 3.8k characters
    fake.answers = [("stream", [content('{"topic": "x"}'), final_chunk(prompt=100)])]
    with pytest.raises(ContextTooLong, match="probably cut"):
        provider_for(fake).generate(LlmRequest("sys", user, SCHEMA, 1000, effort="none"))


def test_count_tokens_from_the_answer_or_from_the_refusal(fake):
    fake.answers = [("stream", [content("S"), final_chunk(prompt=5601, out=1)]),
                    (400, {"error": "request (40141 tokens) exceeds the available context "
                                    "size (32768 tokens), try increasing it"})]
    provider = provider_for(fake)
    assert provider.count_tokens("sys", "S1 text") == 5601
    assert provider.count_tokens("sys", "S1 text") == 40141
    body = fake.chats()[0]
    assert body["options"]["num_predict"] == 1 and body["truncate"] is False
    assert body["think"] is False and "format" not in body


# --- GPU -----------------------------------------------------------------------------------------


def test_partial_gpu_offload_stops_and_unloads(fake):
    fake.size_vram = int(4.1 * GB)
    provider = provider_for(fake)
    expected = r"only partly in GPU memory \(4.1 of 5.6 GB\)"
    with pytest.raises(GpuMemoryShort, match=expected) as info:
        provider.generate(REQUEST)
    assert "qwen3.5:4b" in info.value.remedy and "16384 tokens" in info.value.remedy
    assert fake.requests[-1][2] == {"model": MODEL, "messages": [], "keep_alive": 0}
    assert fake.chats() == []


def test_partial_offload_is_accepted_with_allow_cpu(fake):
    fake.size_vram = int(4.1 * GB)
    fake.answers = [("stream", [content('{"topic": "x"}'), final_chunk()])]
    provider = provider_for(fake, allow_cpu=True)
    provider.generate(REQUEST)
    assert not provider.load_info.on_gpu_entirely


def test_too_little_free_vram_stops_before_loading(fake):
    provider = provider_for(fake, free_mib=2000)
    with pytest.raises(GpuMemoryShort, match="needs about 7.6 GB"):
        provider.generate(REQUEST)
    assert not any(p == "/api/chat" for _, p, _ in fake.requests)


def test_out_of_memory_while_loading(fake):
    fake.load_answer = (500, {"error": "llama runner process has terminated: cudaMalloc failed: "
                                       "out of memory"})
    with pytest.raises(GpuMemoryShort, match="Not enough memory"):
        provider_for(fake).generate(REQUEST)


def test_smaller_context_than_requested_is_refused(fake):
    fake.context_length = 8192
    with pytest.raises(BadRequest, match="8,192-token context instead of 32,768"):
        provider_for(fake).generate(REQUEST)
    assert not fake.loaded


# --- availability and transport ------------------------------------------------------------------


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("exe, expected", [(Path("ollama.exe"), "not running"),
                                           (None, "not installed")])
def test_ollama_not_running_or_not_installed(exe, expected):
    provider = OllamaProvider(MODEL, OllamaOptions(host=f"http://127.0.0.1:{free_port()}"),
                              find_exe=lambda: exe, gpu_query=lambda: None)
    with pytest.raises(OllamaUnavailable, match=expected) as info:
        provider.check()
    assert not info.value.retryable


def test_model_not_downloaded(fake):
    fake.show = (404, {"error": "model 'qwen3.5:9b' not found"})
    with pytest.raises(ModelNotDownloaded) as info:
        provider_for(fake).check()
    assert info.value.remedy == "Run: ollama pull qwen3.5:9b"


def test_check_describes_model_and_version(fake):
    assert provider_for(fake).check() == "qwen3.5:9b (9.0B Q4_K_M), Ollama 0.40.15"


def test_error_inside_the_stream(fake):
    fake.answers = [("stream", [content("{"), {"error": "an error was encountered while "
                                                        "running the model"}])]
    with pytest.raises(ServerError, match="while running the model"):
        provider_for(fake).generate(REQUEST)


def test_stream_that_ends_early_is_a_retryable_network_error(fake):
    fake.answers = [("cut", [content("{")])]
    with pytest.raises(NetworkError) as info:
        provider_for(fake).generate(REQUEST)
    assert info.value.retryable


def test_stalled_answer_times_out(fake):
    fake.answers = [("sleep", 1.0)]
    with pytest.raises(NetworkError, match="sent nothing"):
        provider_for(fake, stall_timeout_s=0.2).generate(REQUEST)


# --- pure helpers --------------------------------------------------------------------------------


@pytest.mark.parametrize("value, url", [
    (None, "http://127.0.0.1:11434"),
    ("", "http://127.0.0.1:11434"),
    ("0.0.0.0", "http://127.0.0.1:11434"),
    ("0.0.0.0:8080", "http://127.0.0.1:8080"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("http://gpu-box:11434", "http://gpu-box:11434"),
    ("https://ollama.example.com", "https://ollama.example.com:443"),
    ("localhost", "http://localhost:11434"),
])
def test_parse_host(value, url):
    assert parse_host(value) == url


def test_is_local():
    assert is_local("http://127.0.0.1:11434") and is_local("http://localhost:11434")
    assert not is_local("http://gpu-box:11434")


def test_kv_cache_of_the_hybrid_qwen35_counts_only_attention_layers():
    # 32 layers, one full-attention layer in 4: 8 layers x 4 KV heads x (256 + 256) x 2 bytes.
    assert kv_bytes_per_token(QWEN_INFO) == 32 * 1024
    dense = {k: v for k, v in QWEN_INFO.items() if "full_attention" not in k}
    assert kv_bytes_per_token(dense) == 128 * 1024
    assert kv_bytes_per_token({}) is None


def test_model_info_from_show():
    info = model_info_from_show(MODEL, QWEN_SHOW)
    assert info.can_think and info.thinking_values == (False, True)
    assert info.family == "qwen35" and info.context_length == 262144
    assert info.describe() == "qwen3.5:9b (9.0B Q4_K_M)"


def test_map_error_too_old():
    error = map_error(412, "this model requires a newer version of Ollama", MODEL, 32768)
    assert isinstance(error, OllamaTooOld) and "Update Ollama" in error.remedy


def test_make_provider_needs_no_key_for_ollama_only():
    provider = make_provider("ollama", None)
    assert isinstance(provider, OllamaProvider) and provider.model == MODEL
    with pytest.raises(MissingKey):
        make_provider("gemini", None)


def test_cloud_providers_refuse_effort_none():
    provider = make_provider("gemini", None, "AIza-test-key-not-real-0000000000000")
    with pytest.raises(BadRequest, match="cannot turn thinking off"):
        provider.generate(REQUEST)


def test_local_model_costs_nothing():
    assert pricing.list_cost("ollama", MODEL, Usage(40_000, 9_000)) == 0.0
    assert pricing.charged_cost("ollama", MODEL, Usage(40_000, 9_000), free_tier=False) == 0.0


def test_parse_nvidia_smi():
    memory = gpu.parse_query("NVIDIA GeForce RTX 3060, 1071, 12288\n")
    assert memory == gpu.GpuMemory("NVIDIA GeForce RTX 3060", 1071, 12288)
    assert memory.free_mib == 11217
    assert gpu.parse_query("") is None and gpu.parse_query("garbage") is None
