# Embedded Voice Agent

A Python application that provides a real-time voice interface using OpenAI's GPT-4o-realtime-preview model. It features a terminal UI for recording and streaming audio to OpenAI, displaying transcripts, and playing back responses with robust audio handling. Built with [Textual](https://www.textualize.io/), it is suitable for embedded and hands-free voice interaction scenarios.

## Features

- **Real-time voice-to-text and response**: Stream audio to OpenAI and receive text and audio responses with minimal latency.
- **Interactive terminal UI**: Visual feedback for recording status and session state.
- **Automatic audio playback**: Responses are played back seamlessly.
- **Session management**: Displays the current OpenAI session and manages reconnections.
- **Keyboard controls**: Start/stop recording and quit via keyboard.
- **Cross-platform audio via sounddevice**.
- **Easily extensible for embedded systems and automation.**

## Requirements

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) for dependency and environment management (recommended)
- Compatible Linux audio stack (PulseAudio, ALSA, etc.)

### Python Dependencies

- [langchain-core](https://pypi.org/project/langchain-core/)
- [numpy](https://pypi.org/project/numpy/)
- [openai](https://pypi.org/project/openai/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [ruff](https://pypi.org/project/ruff/) (development)
- [sounddevice](https://pypi.org/project/sounddevice/)
- [textual](https://pypi.org/project/textual/)
- [textual-dev](https://pypi.org/project/textual-dev/) (development)
- [websockets](https://pypi.org/project/websockets/)

## Quickstart

1. **Clone the repository:**
    ```sh
    git clone https://github.com/sepinedas/embedded-voice-agent.git
    cd embedded-voice-agent
    ```

2. **Install dependencies (recommended):**
    ```sh
    uv pip install -r requirements.txt  # or use 'pyproject.toml' directly with uv
    ```

3. **Set up environment:**
    - Create a `.env` file with your [OpenAI API key](https://platform.openai.com/account/api-keys):
      ```
      OPENAI_API_KEY=your-openai-api-key
      ```

4. **Run the application:**
    ```sh
    uv run app.py
    ```

    Or use the provided Makefile:
    ```sh
    make run
    ```

### Development Mode

To run with live-reload and the Textual developer console:
```sh
make run_dev
```

### Keyboard Controls

- `K`: Start/stop microphone recording
- `Q`: Quit application

## Systemd Service (Linux Startup)

To run the agent automatically on Linux boot and keep it running:

1. Create a systemd service file (e.g., `/etc/systemd/system/embedded-voice-agent.service`):

    ```
    [Unit]
    Description=Embedded Voice Agent
    After=network.target

    [Service]
    User=yourusername
    WorkingDirectory=/path/to/embedded-voice-agent
    ExecStart=/usr/bin/uv run app.py
    Restart=always
    Environment=OPENAI_API_KEY=your-openai-api-key

    [Install]
    WantedBy=multi-user.target
    ```

2. Enable and start the service:
    ```sh
    sudo systemctl daemon-reload
    sudo systemctl enable embedded-voice-agent
    sudo systemctl start embedded-voice-agent
    ```

## File Structure

- `app.py` — Main application logic and UI
- `src/audio.py` — Audio playback and streaming utilities
- `experiment.py` — Experimental features and prototypes
- `pyproject.toml` — Project dependencies and metadata
- `makefile` — Helper commands for running and developing

## License

[MIT](LICENSE)

---

**Note:** This project is under active development. Contributions and issues are welcome!

For more details, visit the [GitHub repository](https://github.com/sepinedas/embedded-voice-agent).
