import asyncio
import base64
from threading import Timer
from typing import Any, cast
import numpy as np

import openwakeword
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime.session import Session
from openwakeword.model import Model
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import RichLog, Static
from typing_extensions import override

from common.audio.audio_player import AudioPlayerAsync
from common.audio.audio_recorder import audio_input_generator

load_dotenv()


DEFAULT_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-4o-realtime-preview"


class SessionDisplay(Static):
    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)"
            if self.is_recording
            else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class RealtimeApp(App):
    CSS_PATH = "app.tcss"

    client: AsyncOpenAI
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None

    def __init__(self) -> None:
        super().__init__()
        self.client = AsyncOpenAI()
        self.connected = asyncio.Event()
        self.should_send_audio = asyncio.Event()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionDisplay(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())
        self.run_worker(self.send_wake_word())

    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(model=DEFAULT_MODEL) as conn:
            self.connection = conn
            self.connected.set()

            await conn.session.update(
                session={"turn_detection": {"type": "server_vad"}}
            )

            acc_items: dict[str, Any] = {}  # noqa: F821

            async for event in conn:
                print(event)

                if event.type == "session.created":
                    self.session = event.session
                    session_display = self.query_one(SessionDisplay)
                    assert event.session.id is not None
                    session_display.session_id = event.session.id
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    continue

                if event.type == "response.audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "response.audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta

                    # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.clear()
                    bottom_pane.write(acc_items[event.item_id])
                    continue

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        status_indicator = self.query_one(AudioStatusIndicator)
        sent_audio = False
        try:
            async for audio_block in audio_input_generator():
                await self.should_send_audio.wait()
                status_indicator.is_recording = True
                connection = await self._get_connection()
                if not sent_audio:
                    asyncio.create_task(connection.send({"type": "response.cancel"}))
                    sent_audio = True

                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(cast(Any, audio_block)).decode("utf-8")
                )

        except KeyboardInterrupt:
            pass

    async def send_wake_word(self) -> None:
        model = Model()
        n_models = len(model.models.keys())
        status_indicator = self.query_one(AudioStatusIndicator)

        try:
            async for audio_block in audio_input_generator(samplerate=16000):
                audio = np.frombuffer(audio_block, dtype=np.int16)
                prediction = model.predict(audio)

                for mdl in model.prediction_buffer.keys():
                    if mdl == "hey_jarvis":
                        scores = list(model.prediction_buffer[mdl])
                        if scores[-1] > 0.5 and not status_indicator.is_recording:
                            self.enable_audio()
                            self.reset_should_send_audio()

        except KeyboardInterrupt:
            pass

    def disable_audio(self):
        status_indicator = self.query_one(AudioStatusIndicator)
        self.should_send_audio.clear()
        status_indicator.is_recording = False

    def enable_audio(self):
        status_indicator = self.query_one(AudioStatusIndicator)
        self.should_send_audio.set()
        status_indicator.is_recording = True

    def reset_should_send_audio(self, delay=3):
        t = Timer(delay, self.disable_audio)
        t.start()

    async def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            if status_indicator.is_recording:
                self.disable_audio()

                if self.session and self.session.turn_detection is None:
                    conn = await self._get_connection()
                    await conn.input_audio_buffer.commit()
                    await conn.response.create()
            else:
                self.enable_audio()


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()
