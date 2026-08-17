# up-and-run-stt — local STT service (GigaAM, CPU) with an OpenAI-compatible API

A local speech-to-text service built on the **GigaAM** model (SberDevices, MIT)
for the **Russian language**, packaged as a single self-contained Docker service
with an **OpenAI-compatible HTTP API**.

The codebase started as a fork of [`haiodo/oaitt`](https://github.com/haiodo/oaitt)
(MIT). Only the **GigaAM Native** engine for CPU was kept; the MLX / WhisperX /
Transformers variants were removed as unnecessary.

> **Russian quality is the priority.** WER/CER measurements and conclusions about
> fitness for use live in [`docs/RU_QUALITY.md`](docs/RU_QUALITY.md). English is
> out of scope (GigaAM v3 is knowingly weaker on it).

---

## Quick start

Nothing to clone and nothing to build — the image is on Docker Hub as
[`vvpreo/up-and-run-stt`](https://hub.docker.com/r/vvpreo/up-and-run-stt), for
both `linux/amd64` and `linux/arm64`:

```bash
docker run -d --name up-and-run-stt \
  -p 9007:9007 \
  -v gigaam-models:/app/data \
  -e AUTH_TOKEN=change-me \
  -e GIGAAM_MODELS=v3_e2e_ctc \
  -e DEFAULT_LANGUAGE=ru \
  -e MODEL_IDLE_TIMEOUT=0 \
  -e VAD_CHUNKING=true \
  -e MAX_UPLOAD_MB=200 \
  -e MAX_PENDING_REQUESTS=8 \
  -e LOG_LEVEL=INFO \
  --restart unless-stopped \
  vvpreo/up-and-run-stt:latest
```

What the flags do, since only the first three are strictly required:

| Flag | Why it matters |
|---|---|
| `-v gigaam-models:/app/data` | **Do not skip this.** Model weights are not baked into the image — they are downloaded on first start. Without a volume they land inside the container and are re-downloaded from scratch every time it is recreated. |
| `-e AUTH_TOKEN=change-me` | Enables Bearer authorization on the transcription endpoints. Leave it unset and **the API is open to anyone who can reach the port**. |
| `-e GIGAAM_MODELS=v3_e2e_ctc` | Which models the instance serves, comma-separated; the first is the default. Each costs ~1.4 GB of RAM. `v3_e2e_rnnt` is more accurate and slower — see [Performance](#performance-on-this-cpu). |
| `-e MODEL_IDLE_TIMEOUT=0` | Never unload the model, so the first request after a pause is not slow. |
| `-e VAD_CHUNKING=true` | Split long audio at speech pauses instead of on a fixed grid, and skip silence. |
| `-e MAX_UPLOAD_MB` / `MAX_PENDING_REQUESTS` | Upload cap (`413` above it) and concurrency cap (`429` above it). |

Every value shown above is already the default, so the short form below behaves
identically — they are spelled out to show what is worth changing. The full list
is in [Configuration](#configuration-environment-variables).

```bash
docker run -d -p 9007:9007 -v gigaam-models:/app/data \
  -e AUTH_TOKEN=change-me vvpreo/up-and-run-stt:latest
```

**First start downloads the model weights** (~845 MB for one fp32 model) from
[vvpreo/gigaam-v3-onnx](https://huggingface.co/vvpreo/gigaam-v3-onnx), so it takes
a few minutes; every later start is instant because the volume persists. Follow it
with `docker logs -f up-and-run-stt`.

Readiness check:

```bash
curl -s http://localhost:9007/health | python3 -m json.tool
```

Expected: `"model_loaded": true`, `"engine": "gigaam"`.

First transcription:

```bash
curl -s http://localhost:9007/v1/audio/transcriptions \
  -H "Authorization: Bearer change-me" \
  -F "file=@audio.ogg" -F "response_format=text"
```

Available tags: `latest`, plus `X.Y.Z` / `X.Y` / `X` — pin as tightly as you like.
The image is **~0.7 GB** (inference on ONNX Runtime, no PyTorch).

### Running from source

Only needed if you intend to modify the service. `docker-compose.yml` builds the
image locally, and all configuration lives in its `environment` block, each
variable commented; there is no separate `.env`.

```bash
docker compose up -d          # build (first time) + run
docker compose logs -f
```

Dependencies are managed with [uv](https://docs.astral.sh/uv/): `pyproject.toml`
declares them and `uv.lock` pins the exact resolution, including transitive
packages. The lock is universal — one resolution covers both `linux/amd64` and
`linux/arm64`, so both architectures of the published image install identical
versions. The Docker build runs `uv sync --frozen`, which fails rather than
silently re-resolving if the lock has drifted from `pyproject.toml`. To upgrade:
`uv lock --upgrade`, rebuild, run the tests.

The weights mirror is a conversion of the official checkpoints
(`scripts/convert_onnx.py`); override it with `GIGAAM_ONNX_BASE_URL` to serve them
from your own host. The PyTorch variant, used for weight conversion and quality
cross-checks, is `Dockerfile.torch`.

The service listens on `0.0.0.0:9007`, so it is reachable from other hosts on the
network and from other containers. It comes back up after a reboot
(`restart: unless-stopped`).

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible (field `file`). For the OpenAI SDK and any client that speaks the Whisper API. |
| `POST` | `/stt/asr` | Native (field `audio_file`), richer response: segments, words, metrics. |
| `POST` | `/stt/emotion` | Speech emotion (GigaAMEmo): angry / sad / neutral / positive. Not part of the OpenAI contract, which has no notion of emotion — hence `/stt/`. |
| `WS` | `/stt/stream` | Live audio input: push PCM while speaking, get phrases back. A native protocol, not OpenAI's Realtime API. `GET` the same path for the spec. |
| `GET` | `/v1/models`, `/v1/models/{id}` | Models available on this instance (OpenAI format, for GUI clients). |
| `POST` | `/v1/audio/translations` | Not supported (the model is Russian-only) — returns a proper 400. |
| `GET`  | `/health` | Status, models, queue, feature flags, memory. |
| `GET`  | `/` | WebUI console: log in with `AUTH_TOKEN`, mic/file input, both API contracts, all formats, word timestamps, emotions ([src/static/index.html](src/static/index.html)). |

A Swagger UI is also available at `http://localhost:9007/docs`.


### `POST /v1/audio/transcriptions` (OpenAI-compatible)

The file field is **`file`** (same as OpenAI). `response_format`: `json` (default),
`text`, `verbose_json`, `srt`, `vtt`.

```bash
# json → {"text": "..."}
curl -s http://localhost:9007/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=json"

# text → plain text
curl -s http://localhost:9007/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "response_format=text"
```

### `POST /stt/asr` (native)

The file field is **`audio_file`**. Query parameters: `output`
(`json`/`text`/`srt`/`vtt`/`tsv`), `language`, `word_timestamps`, `model`,
`vad` (VAD chunking, defaults to `VAD_CHUNKING`).

```bash
curl -s "http://localhost:9007/stt/asr?output=json&language=ru" \
  -F "audio_file=@audio.wav"
```

The response (`output=json`) contains `text`, `language`, plus `segments` for long
audio, as well as `confidence` / `chars_per_second` (diagnostics).

### SSE streaming (`stream=true`) — response only

> **Read this before designing a client.** What is streamed here is the
> **response**, not the request. The audio still goes up as one complete HTTP
> request; only the text comes back incrementally. To stream the *input* — to
> feed a live microphone into an open connection — use the WebSocket at
> [`/stt/stream`](#live-audio-input--websocket-sttstream) instead.

This mirrors OpenAI exactly: their `/v1/audio/transcriptions` with `stream=true`
also streams the response for an already-uploaded file. OpenAI's own streaming
*input* lives in a separate product, the Realtime API over WebSocket, whose
protocol this service does **not** implement — `/stt/stream` is a native
extension with its own protocol, which is why it sits under `/stt/` rather than
`/v1/`.

How it works: `-F "stream=true"` → `text/event-stream` with
`transcript.text.delta` events (emitted as the server finishes each chunk) and a
final `transcript.text.done` (full text + `usage`). Works with the OpenAI SDK
(`stream=True`). On a long file the first text arrives in ~1 s instead of waiting
for the complete response. Stream chunk size is `STREAM_CHUNK_SEC` (default 12 s);
dropping the connection cancels inference.

```bash
curl -sN http://localhost:9007/v1/audio/transcriptions \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -F "file=@audio.ogg" -F "stream=true"
```

**Practical consequence:** for a file you already have, text starts appearing
almost immediately. For a live microphone, nothing can be transcribed until the
recording is finished and sent — the model is offline (it needs a complete
segment) and the transport has no way to deliver a partial request body. A client
that wants text *while* the user is still speaking has to cut the audio itself and
send each piece as a separate request, which is an application-level workaround,
not a feature of this API.

### Live audio input — `WebSocket /stt/stream`

This is the other half: here the **request** streams. You push raw PCM while the
person is still speaking and get text back phrase by phrase, without waiting for
the recording to end.

```
ws://localhost:9007/stt/stream?token=<AUTH_TOKEN>&language=ru
```

The client is deliberately dumb: it sends frames and decides nothing. Where a
phrase ends, what counts as silence and what to discard is decided on the server
by silero-vad — the same neural VAD that chunks uploaded files, so there is one
implementation of that logic rather than two.

Send raw **PCM s16le, 16 kHz, mono** as binary frames (any size; 50–200 ms is
sensible). Compressed input is not accepted on purpose: it would need a decoder
process per connection, while raw PCM costs 256 kbit/s and no CPU. Control
messages are JSON: `{"type":"commit"}` closes the current phrase immediately,
`{"type":"close"}` ends the session.

The server replies with JSON events: `session.created`, `speech.started` /
`speech.stopped`, `transcript.text.delta` for each finished phrase, a final
`transcript.text.done`, and `stream.overflow` if inference falls behind and the
oldest queued phrase had to be dropped.

**Results arrive per phrase, never word by word.** GigaAM is an offline model — it
needs a complete segment — so there are no partial hypotheses inside a phrase.
Latency from the end of a phrase to its text is about a second: ~600 ms to confirm
the pause plus inference (~0.3 s for a 5-second phrase).

Measured cost on the reference CPU: an open session is **~0.5% of a core** for
continuous voice detection plus ~2 MB of buffers, and inference runs only when a
phrase closes — about **5% of a core** for one continuously speaking user. That is
why the session limit (`STREAM_MAX_SESSIONS`, default 32) is much higher than the
limit on concurrent file transcriptions (`MAX_PENDING_REQUESTS`, default 8): an
open stream is nearly free, only phrase completions cost anything.

Silence is trimmed before inference. On a 137-second sample that cut the audio fed
to the model from 137 s to 69 s and, unexpectedly, **improved** accuracy — long
stretches of silence make the model produce filler.

A bad token fails the handshake with **HTTP 403** and no socket is opened. Hitting
the session limit is different on purpose: the socket opens, you get an `error`
event with `retry: true`, then close code **1013** — so a client can tell "not
allowed" from "come back later".

Full protocol, including a runnable Python client, is at
**`GET /stt/stream`** and in Swagger under the *Streaming* tag. It is a
regular JSON endpoint because OpenAPI cannot describe a WebSocket.

### Microphone in the WebUI

The console makes the endpoint an explicit choice, because there are four of
them and they are not variations of one another:

| Выбор | Ручка | Что делает |
|---|---|---|
| OpenAI | `POST /v1/audio/transcriptions` | Whisper-compatible. The only one with `stream=true` (SSE response streaming). |
| Нативная | `POST /stt/asr` | Our own; richer response. Ignores `stream`. |
| Живая диктовка | `WS /stt/stream` | Input streams; phrases come back while you speak. Microphone only. |
| Эмоции | `POST /stt/emotion` | Additive — runs in parallel with any of the above. |

The first three are mutually exclusive and picked from one selector; emotions is
a checkbox because it is a separate parallel request, not a mode.

This matters because "streaming" meant two unrelated things and one checkbox used
to cover both. `stream=true` is a **parameter** of the OpenAI endpoint that
changes how the response comes back; the live WebSocket is a **different endpoint**
that changes how audio goes in. They are now in different places in the UI, and
each endpoint shows only the options that belong to it — response format and word
timestamps disappear for the live endpoint, `stream=true` appears only under
OpenAI, and a line under the controls states what is in effect.

Audio goes in through one block: drop a file, click the frame to pick one, or
press **🎙 Начать запись**. With the live endpoint selected the file half is
disabled, since a WebSocket session takes a microphone rather than a file.

### Supported audio input formats

Practically every common format is accepted: **wav, flac, mp3, ogg/vorbis, opus,
m4a/aac, webm, wma** and others. Decoding is two-stage: `libsndfile` first
(wav/flac/ogg/opus/mp3 — the fast path), everything else falls back to the static
`ffmpeg` binary shipped in the image; resampling uses `scipy` (polyphase). Raw
headerless PCM bytes are not accepted — a container is required (WAV at minimum).
The input file's sample rate and channel count do not matter: everything is
converted to 16 kHz mono automatically.

The effect of format on speed is **negligible**. Measured on the same 137 s clip
(i7-8750H, `v3_e2e_ctc`, best of several runs):

| Format | Size | Decode only | Full request (localhost) |
|---|---|---|---|
| m4a | 1.2 MB | 89 ms | 6.5 s |
| wma | 2.5 MB | 113 ms | 6.9 s |
| wav | 12.9 MB | 157 ms | 6.4 s |
| mp3 | 1.0 MB | 213 ms | 6.7 s |
| flac | 5.5 MB | 224 ms | 6.3 s |
| ogg | 0.8 MB | 241 ms | 6.4 s |
| webm | 1.1 MB | 266 ms | 6.8 s |
| opus | 1.1 MB | 631 ms | 6.9 s |

Decoding takes 0.09–0.6 s against ~6 s of inference (< 10% of the request); the
spread in total time across formats is comparable to the noise between runs.
Practical takeaway: **choose the format by transfer size, not by decode speed** —
over the network, compressed opus/mp3 (~1 MB) beats wav (~13 MB); locally it makes
no difference.

---

## Configuration (environment variables)

Everything is configured through environment variables at runtime — in the
`environment` block of `docker-compose.yml` (comments included there) or via
`docker run -e`. 

### Models: the instance set and per-request selection

An instance serves the set of models listed in `GIGAAM_MODELS` (for example,
`v3_e2e_ctc,v3_e2e_rnnt`): all of them are loaded at startup (weights are fetched
into the volume as needed) and kept warm in RAM — **~1.4 GB each**.

The model set and the default are visible in `/health` (`models`, `default_model`).
With `MODEL_WORKERS > 1` the pool serves the default model only.

#### Model variants

| Value | Punctuation | Notes |
|---|---|---|
| `v3_e2e_ctc` | yes | **Default.** End-to-end, text normalization, fastest. |
| `v3_e2e_rnnt` | yes | End-to-end, claimed best quality, slower. |
| `v3_ctc` | no | No punctuation, fast — a fallback for interactive dictation. |
| `v3_rnnt` | no | No punctuation, RNNT. |

Changing the set **without rebuilding the image**: edit `GIGAAM_MODELS` in
`docker-compose.yml`, then run `docker compose up -d --force-recreate`.

---

## Model persistence

Weights are **not baked into the image** and **not re-downloaded on every start**.
They live in the named Docker volume `gigaam-models`, mounted at `/app/data`
(`MODEL_CACHE_DIR`). They are downloaded once, on first use of a model, and survive
restarts and image rebuilds.

- Inside the container: `/app/data/onnx/<model>.onnx` (+ `.yaml`); tokenizers for
  the e2e models are at `/app/data/gigaam/<model>_tokenizer.model`.
- Weights source: HF Hub [vvpreo/gigaam-v3-onnx](https://huggingface.co/vvpreo/gigaam-v3-onnx)
  (override with `GIGAAM_ONNX_BASE_URL`); tokenizers come from the SberDevices CDN.
  All over plain HTTPS, no tokens involved.
- Size: an fp32 model is ≈ 845 MB (int8 variants ≈ 215 MB — `GIGAAM_ONNX_VARIANT=.int8`).

Inspect the volume contents:

```bash
docker run --rm -v gigaam-models:/data alpine ls -lh /data/gigaam
```

> The container runs as an unprivileged user (uid 1000). If the weights volume was
> created by an older (root) version of the image, run this once:
> `docker run --rm -v gigaam-models:/data alpine chown -R 1000:1000 /data`.

---

## Performance (on this CPU)

Host: **Intel Core i7-8750H** (6 cores / 12 threads, 2.2 GHz), 64 GB RAM, no GPU.

<!-- PERF_TABLE_START -->
Measured on 30 clips from FLEURS `ru_ru` (6.2 min of clean read speech). Details,
error analysis and caveats are in [`docs/RU_QUALITY.md`](docs/RU_QUALITY.md).

| Model | WER | CER | RTF | Speed | RAM (loaded) |
|---|---|---|---|---|---|
| `v3_e2e_rnnt` | **4.9 %** | 1.3 % | 0.111 | 9.0× | 1.44 GiB |
| `v3_e2e_ctc` *(default)* | 6.2 % | 1.3 % | 0.088 | 11.3× | 1.41 GiB |
| `v3_ctc` | 8.3 % | 2.2 % | 0.091 | 10.9× | 1.42 GiB |

**Recommendation:** for the best Russian quality use `v3_e2e_rnnt` (there is plenty
of speed headroom on CPU: a 5–15 s clip is processed in ~0.5–1.7 s). The default
`v3_e2e_ctc` is slightly faster and nearly as accurate.
<!-- PERF_TABLE_END -->

**RTF** (realtime factor) = processing time / audio duration; lower is better
(`0.1` ≈ 10× faster than realtime). For short dictation clips (2–15 s), latency is
RTF·duration plus network overhead.

### Scaling across cores (single worker)

Inference speed scales **non-linearly** with cores — returns diminish quickly
(small ops do not parallelize, and the cores contend for the shared memory bus).
Measurement: the same i7-8750H, `v3_e2e_ctc`, a 20 s clip,
`torch.set_num_threads(N)`, best of 3 runs:

| CPU threads | Time | Speedup | Speed | Per-core efficiency |
|---|---|---|---|---|
| 1 | 2.37 s | ×1.0 | 8.5× | 8.5× |
| 2 | 1.37 s | ×1.73 | 14.6× | 7.3× |
| 3 | 1.05 s | ×2.26 | 19.0× | 6.3× |
| 4 | 0.90 s | ×2.63 | 22.3× | 5.6× |
| 6 | 0.83 s | ×2.86 | 24.2× | 4.0× |

Practical conclusions for tuning `MODEL_WORKERS` × `OMP_NUM_THREADS`:

- **Minimum latency for a single request** (interactive dictation, one user):
  1 worker with all cores — the current default. Note that 3–4 cores already give
  almost the same result as 6.
- **Maximum aggregate throughput** (several concurrent clients): several workers
  with 2–3 cores each is a better deal — for example, 3 workers × 2 threads add up
  to ~44× realtime versus 24× for "1 worker × 6 threads" on the same cores. The
  price: ~1.4 GB of extra RAM per additional worker (its own copy of the model) and
  a slower individual request; the real total is slightly below the theoretical one
  because of the shared memory bus.

On different hardware the absolute numbers will differ, but the shape of the curve
(diminishing returns past 2–4 cores) is typical for CPU inference with torch.

---

## Integration

### Desktop dictation clients

Any dictation app that lets you point it at a custom OpenAI-compatible endpoint
works without adaptation. Typical settings:

- **API Endpoint:** `http://<service-address>:9007/v1/audio/transcriptions`
- **API Key:** the `AUTH_TOKEN` value the service was started with (leave empty if
  authorization is disabled)
- **Model Name:** `whisper-1` — or any name from `GIGAAM_MODELS` to pick a
  specific model

Two things to check if it does not connect: the scheme is `http://` unless you put
the service behind a TLS-terminating proxy, and the host must actually be reachable
from the client machine (same LAN — direct IP; a remote host — an SSH port forward
such as `-L 9007:localhost:9007`).

### OpenAI Python SDK

The service is a drop-in replacement for the OpenAI Whisper API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://<service-address>:9007/v1/", api_key="<AUTH_TOKEN>")

with open("audio.wav", "rb") as f:
    r = client.audio.transcriptions.create(
        model="whisper-1",          # ignored
        file=f,
        language="ru",
        response_format="json",     # or "text", "verbose_json"
    )
print(r.text)
```

For plain `curl`, see the [API](#api) section above (both `/v1/...` and `/stt/asr`).

---

## Russian quality evaluation

WER/CER measurements plus RTF and RAM on a Russian dataset, and a comparison of the
model variants, are in [`docs/RU_QUALITY.md`](docs/RU_QUALITY.md). The benchmark
tooling was removed from the project (it was trimmed down to a working minimum);
the results are preserved in the report.

---

## Operations

```bash
docker compose up -d / down / logs -f / ps       # the service
./build.sh                                       # build a local image
./build.sh myuser/up-and-run-stt --push              # build + publish to Docker Hub
curl -s localhost:9007/health | python3 -m json.tool   # status

# quick run against the bundled sample (token — from docker-compose.yml):
curl -s localhost:9007/v1/audio/transcriptions \
  -H "Authorization: Bearer <AUTH_TOKEN>" \
  -F "file=@sample-data/Sobolev_Andrey_1_0_00-2_17.ogg" -F "response_format=text"
```

## Tests

The integration tests talk to a running service (`docker compose up -d`, then wait
for `model_loaded: true`):

```bash
uv sync --group dev
uv run pytest tests/ -v
```

Coverage: /health, authorization, both endpoints, the model registry, every input
format, the size limit (`413`), backpressure (`429`), and event-loop liveness during
transcription. To point them at a different service address, use `STT_BASE_URL` /
`STT_AUTH_TOKEN`.

## Troubleshooting

- **`/health` stays at `model_loaded: false` for a long time** — the first start
  downloads the weights; watch `docker compose logs -f`. The healthcheck
  `start_period` is 300 s.
- **Not reachable from another machine** — check that port `9007` is visible from
  the client (LAN/SSH), that you are using `http://`, and that the firewall is not
  blocking it.
- **Slow on long audio** — that is the CPU; for interactive dictation use short
  clips and, if needed, `v3_ctc` (faster, no punctuation).

## Licenses and credits

- The project is based on the work of **Andrey Sobolev**
  ([haiodo/oaitt](https://github.com/haiodo/oaitt)) — the original author of the
  OAITT server. This fork and `haiodo/oaitt` are MIT (see `LICENSE`; the Andrey
  Sobolev copyright is preserved).
- `gigaam` (vendored under `vendor/gigaam`) — MIT, SberDevices.
