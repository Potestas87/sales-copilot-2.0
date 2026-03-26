# Sales Copilot

[![Tests](https://github.com/Potestas87/sales-copilot-2.0/actions/workflows/tests.yml/badge.svg)](https://github.com/Potestas87/sales-copilot-2.0/actions/workflows/tests.yml)

A real-time AI sales assistant that listens to live sales calls, transcribes both speakers, and surfaces suggested rebuttals and responses in real time — powered by self-hosted Whisper and Mistral 7B running on a cloud GPU.

Built as a DevOps portfolio project demonstrating containerized GPU workloads, CI/CD pipelines, and real-time ML inference at the edge.

## How It Works

```
┌─────────────────── Mac Client ───────────────────┐      ┌────────────── RunPod GPU Server ──────────────┐
│                                                   │      │                                               │
│  Microphone (salesperson) ──┐                     │      │                                               │
│                             ├─→ VAD ─→ WebSocket ─┼─ WSS ┼─→ faster-whisper ─→ Mistral 7B ─→ JSON reply │
│  BlackHole (customer call) ─┘                     │      │       (transcription)     (inference)         │
│                                                   │      │                                               │
│  ◀── Suggestion Display ◀── WebSocket ◀───────────┼──────┼── intent + suggestion + confidence            │
└───────────────────────────────────────────────────┘      └───────────────────────────────────────────────┘
```

1. **Audio capture** — The Mac client captures two audio streams simultaneously: the salesperson's microphone and the customer's call audio routed through BlackHole (a virtual audio device).
2. **Voice activity detection** — Each stream passes through a silero-vad filter that detects when someone is speaking and buffers complete utterances.
3. **Streaming to server** — Completed utterances are base64-encoded, labeled with the speaker (`customer` or `salesperson`), and sent as JSON over a persistent WebSocket connection to the GPU server on RunPod.
4. **Transcription** — The server transcribes audio using faster-whisper (CTranslate2-optimized Whisper, large-v3 model).
5. **Intent classification and response generation** — Customer utterances are passed to Mistral 7B (4-bit quantized GGUF) along with rolling conversation context (last 20 turns). The LLM classifies the utterance as an objection, question, buying signal, or none, and generates a suggested response based on the sales playbook. Salesperson utterances are transcribed for context but don't trigger inference.
6. **Display** — The client renders the suggestion with intent type, confidence score, and reasoning on screen in real time.

## Project Structure

```
sales-copilot/
├── client/                       # Mac-side Python application
│   ├── main.py                   #   Orchestrator — wires audio, VAD, WS, display
│   ├── audio_capture.py          #   Dual-stream audio capture (mic + BlackHole)
│   ├── vad.py                    #   Voice activity detection (silero-vad)
│   ├── websocket_client.py       #   Async WebSocket client with reconnection
│   ├── protocol.py               #   Client-side Pydantic message models
│   └── display.py                #   Real-time suggestion display
├── server/                       # GPU server (FastAPI + AI models)
│   ├── main.py                   #   Uvicorn entry point
│   ├── api.py                    #   WebSocket endpoint + session memory
│   ├── transcription.py          #   faster-whisper wrapper
│   ├── inference.py              #   Mistral 7B suggestion engine
│   ├── protocol.py               #   Server-side Pydantic message models
│   └── prompts.py                #   System/user prompt builder from playbook
├── config/
│   └── sales_playbook.yaml       # Product info, objection handling, tone
├── docker/
│   ├── Dockerfile                # CUDA 12.8 devel image, compiles llama-cpp-python from source
│   └── docker-compose.yml
├── scripts/
│   ├── entrypoint.sh             # Downloads model on first boot, starts server
│   ├── download_models.sh
│   └── test.sh
├── tests/
│   ├── test_protocol_models.py   # Protocol validation tests
│   ├── test_inference_parsing.py # LLM response parsing tests
│   └── test_websocket_client_integration.py
├── .github/workflows/
│   ├── tests.yml                 # CI: pytest on every push/PR
│   └── docker-runpod.yml         # CD: manual Docker build + push to Docker Hub
├── requirements-client.txt
├── requirements-server.txt
├── requirements-test.txt
├── .env.example
└── README.md
```

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Audio capture | sounddevice + BlackHole | Dual-stream capture (mic + call audio) |
| Voice activity detection | silero-vad | Filters silence, buffers complete utterances |
| Transport | WebSockets (WSS) | Low-latency bidirectional streaming via RunPod proxy |
| Server framework | FastAPI + uvicorn | Async WebSocket handler with session state |
| Transcription | faster-whisper (large-v3) | CTranslate2-optimized Whisper, ~2x faster than OpenAI |
| Language model | Mistral 7B Instruct v0.2 (Q4_K_M) | 4-bit quantized GGUF, ~4.5 GB, full GPU offload |
| LLM runtime | llama-cpp-python | Compiled from source with CUDA for GPU acceleration |
| Containerization | Docker (CUDA 12.8 devel) | Reproducible GPU builds with nvcc included |
| GPU hosting | RunPod (RTX 5090) | On-demand cloud GPU with persistent storage |
| CI/CD | GitHub Actions | Automated testing + manual Docker publish |

## Prerequisites

- **Mac** with Python 3.11+
- **BlackHole** virtual audio driver (routes call audio to the client)
- **Docker** with buildx support (for building the server image)
- **RunPod account** with an SSH key configured
- **Docker Hub account** (for pushing the server image)

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/Potestas87/sales-copilot-2.0.git
cd sales-copilot-2.0
cp .env.example .env
```

Edit `.env` with your RunPod pod URL and audio device names:

```env
SERVER_URL=wss://<your-pod-id>-8000.proxy.runpod.net/ws
MIC_DEVICE_NAME=SoloCast        # your microphone
CALL_DEVICE_NAME=BlackHole      # virtual audio device for call audio
WHISPER_MODEL=large-v3
```

### 2. Install client dependencies

```bash
pip install -r requirements-client.txt
```

### 3. Build and push the server image

```bash
docker buildx build --platform linux/amd64 \
  -t <your-dockerhub>/sales-copilot:latest \
  -f docker/Dockerfile . --push
```

This compiles llama-cpp-python from source with CUDA support inside the image (~10-15 min build).

### 4. Deploy on RunPod

1. Create a GPU pod (RTX 4090/5090 recommended) with your Docker image
2. Attach a persistent volume at `/workspace` (stores the Mistral model between restarts)
3. Set environment variable: `LLM_MODEL_PATH=/workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf`
4. The entrypoint script auto-downloads the model (~4.5 GB) on first boot

### 5. Run the client

```bash
python client/main.py
```

## Sales Playbook

The LLM's behavior is configured through `config/sales_playbook.yaml` — no code changes needed. The playbook defines:

- **Product info** — name, description, value propositions
- **Objection handling** — specific response angles for price, timing, competitor, complexity, and authority objections
- **Buying signal guidance** — how to respond when the customer shows interest
- **Tone** — the conversational style for generated suggestions

Edit the YAML to match your product and rebuild the Docker image to deploy changes.

## CI/CD

- **Tests** (`tests.yml`) — Runs `pytest` on every push and pull request. Covers protocol validation, LLM response parsing, and WebSocket client integration.
- **Docker publish** (`docker-runpod.yml`) — Manually triggered workflow that builds the Docker image and pushes to Docker Hub. Requires `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets.

## WebSocket Protocol

The client and server communicate over a single persistent WebSocket connection using JSON envelopes.

**Client → Server** (utterance):
```json
{
  "type": "utterance",
  "speaker": "customer",
  "sample_rate": 16000,
  "ts_ms": 1711500000000,
  "audio_b64": "<base64-encoded float32 audio>"
}
```

**Server → Client** (inference):
```json
{
  "type": "inference",
  "speaker": "customer",
  "transcript": "I'm not sure we can afford this right now",
  "intent": "objection",
  "suggestion": "That's a valid concern. Our average customer saves 8 hours per rep per week...",
  "reasoning_short": "Price objection — reframe around ROI",
  "confidence": 0.85,
  "latency_ms": 1200.5
}
```

## License

MIT
