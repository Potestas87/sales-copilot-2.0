# Sales Copilot

A real-time AI sales assistant that transcribes live sales calls and surfaces suggested responses whenever a customer raises an objection or question — powered by self-hosted Whisper and Mistral 7B on a cloud GPU.

## Architecture

- **Client (Mac):** Captures microphone + call audio, streams to server via WebSocket, displays AI suggestions on screen
- **Server (RunPod GPU):** Runs faster-whisper for transcription and Mistral 7B for objection detection and response generation
- **Transport:** WebSockets for low-latency bidirectional communication

## Project Structure

```
sales-copilot/
├── client/               # Mac-side Python app
├── server/               # GPU server (FastAPI + AI models)
├── config/               # Sales playbook and settings
├── docker/               # Dockerfile and compose config
├── requirements-client.txt
├── requirements-server.txt
└── .env.example          # Environment variable template
```

## Setup

See full setup instructions in `Sales-Copilot-Project-Overview.md`.

1. Copy `.env.example` to `.env` and fill in your values
2. Install client dependencies: `pip install -r requirements-client.txt`
3. Deploy server to RunPod using the provided Dockerfile
4. Run the client: `python client/main.py`

## Tech Stack

| Layer | Technology |
|---|---|
| Audio capture | sounddevice + BlackHole |
| Voice activity detection | silero-vad |
| Transport | WebSockets |
| Server framework | FastAPI |
| Transcription | faster-whisper |
| Language model | Mistral 7B (4-bit quantized) |
| Containerization | Docker |
| GPU hosting | RunPod |
