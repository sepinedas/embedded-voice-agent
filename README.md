# Embedded Voice Agent (Raspberry Pi)

OpenAI ReAct voice agent targeting Raspberry Pi hardware. This project records microphone audio, streams it to OpenAI's Realtime model, and plays model-generated audio back — using GPIO for a wake button and LEDs to indicate state.

## Highlights

- Realtime voice interaction using OpenAI Realtime API (model: gpt-realtime).
- Simple GPIO-based wake button and LED indicators (pins 15, 17, 27).
- Uses sounddevice/PyAudio for audio I/O and a tiny playback queueing player.
- Designed for small Raspberry Pi projects and experimentation.

## Stack

- **Language(s):** Python 3.12+
- **Framework / runtime:** plain asyncio + sounddevice + OpenAI realtime client
- **Notable libraries:** openai (realtime), sounddevice, numpy, pyaudio, gpizero

## Repository layout

```
.
├── main.py                  # Application entrypoint: Realtime loop, GPIO, audio I/O
├── pyproject.toml           # Project metadata & dependency list
├── uv.lock                  # Lock file (package manager-specific)
├── .gitignore
└── common/
    ├── __init__.py
    └── audio/
        ├── __init__.py
        ├── audio_player.py  # Audio playback queue / player implementation
        └── audio_recorder.py# Audio input generator (PCM16 chunk producer)
```

## How it fits together

- `main.py` creates a `RealtimeApp` that:
  - manages a connection to OpenAI's realtime endpoint (AsyncOpenAI.realtime.connect),
  - records audio with `common/audio/audio_recorder.audio_input_generator` and pushes PCM16 chunks to the realtime connection,
  - receives audio and text deltas from the model and passes audio frames to `common/audio/audio_player.AudioPlayerAsync` for playback,
  - uses `gpizero` InputDevice and LED objects to observe a physical button and toggle LEDs (wake/connected).
- Audio/frame handling and buffering lives in `common/audio/*` while control logic and the realtime event loop are in `main.py`.

## Requirements & hardware

- A Raspberry Pi (or Linux machine with GPIO access) and microphone + speaker.
- GPIO pins used by the code:
  - connected_led = GPIO 27
  - wake LED = GPIO 17
  - input (button) = GPIO 15 (pull_up disabled)
- OpenAI API access with Realtime privileges for the voice model (gpt-realtime).
- Python 3.12 (pyproject.toml declares >=3.12).

## Python dependencies (from pyproject.toml)

- openai >= 1.107.1
- sounddevice >= 0.5.2
- numpy >= 2.3.3
- pyaudio >= 0.2.14
- python-dotenv >= 1.1.1
- gpizero >= 2.0.1
- lgpio >= 0.2.2.0
- websockets >= 1.5.0

## Quickstart — minimal steps to run

1. Clone and enter repo
   ```
   git clone https://github.com/sepinedas/embedded-voice-agent.git
   cd embedded-voice-agent
   ```

2. (Recommended) Create and activate a virtual environment
   ```
   python3 -m venv venv
   source venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. Install the runtime dependencies
   ```
   python -m pip install openai sounddevice numpy pyaudio python-dotenv gpizero lgpio websockets
   ```
   Note: installing pyaudio and sounddevice on Raspberry Pi may require system packages (e.g., libportaudio-dev).

4. Provide your OpenAI API key
   - Create a `.env` file in the repo root with:
     ```
     OPENAI_API_KEY=sk-...
     ```

5. Wire hardware
   - LED (connected) to GPIO 27
   - LED (wake) to GPIO 17
   - Button/input to GPIO 15 (pull_up disabled)

6. Run the app
   ```
   python main.py
   ```

## Configuration & customization

- Model and voice: `main.py` currently sets:
  - DEFAULT_MODEL = "gpt-realtime"
  - VOICE = "coral"
- GPIO pins are defined in `main.py` as `LED(27)`, `LED(17)`, `InputDevice(15)`.
- Audio framing, chunk size and sample format are managed in `common/audio/*`.

## Files of interest

- `main.py` — main runtime loop, Realtime connection handling, GPIO handling.
- `common/audio/audio_player.py` — playback queue and frame consumer.
- `common/audio/audio_recorder.py` — audio input generator used to create PCM16 chunks sent to the API.

## Troubleshooting & tips

- Microphone / sounddevice:
  - Add a small script using `sounddevice.query_devices()` to list audio devices if needed.
  - On Raspberry Pi, install portaudio system packages before pip-installing pyaudio.
- Permissions:
  - Running sound and accessing audio devices may require appropriate user permissions.
- OpenAI access:
  - Realtime voice models may require special access and an up-to-date `openai` package.

## Development notes

- Core concurrent tasks:
  - `handle_realtime_connection()` — reads events and handles model deltas
  - `send_mic_audio()` — produces audio chunks and sends them as `input_audio.buffer` events
  - `handle_button()` — watches the GPIO input to toggle wake/sleep states

## Contributing

- For code changes, open issues describing the change (include device/OS details) and I can propose patches.

## Try asking

- How do I change the GPIO pins for the wake button and LEDs in main.py?
- How can I switch to a different realtime model or change the VOICE value (where in main.py)?
- Where can I tune the audio buffer size or frames-per-chunk used by the recorder/player (which files and variables in common/audio/)?

## Acknowledgements

- Built with openai realtime client, sounddevice, gpizero and minimal helper modules.
