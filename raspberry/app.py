import asyncio
from threading import Timer
import numpy as np
from dotenv import load_dotenv
from openwakeword.model import Model
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from typing import Any, cast
import base64

from common.audio.audio_player import AudioPlayerAsync
from common.audio.audio_recorder import audio_input_generator

load_dotenv()


DEFAULT_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-4o-realtime-preview"


class RealtimeApp:
    client: AsyncOpenAI
    connection: AsyncRealtimeConnection | None
    connected: asyncio.Event
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    is_recording: bool

    def __init__(self) -> None:
        self.client = AsyncOpenAI()
        self.connected = asyncio.Event()
        self.should_send_audio = asyncio.Event()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.is_recording = False

    async def run(self):
        print("running")
        await asyncio.gather(
            self.handle_realtime_connection(),
            self.send_mic_audio(),
            self.send_wake_word(),
        )

    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(model=DEFAULT_MODEL) as conn:
            self.connection = conn
            self.connected.set()

            acc_items: dict[str, Any] = {}  # noqa: F821

            async for event in conn:
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

                    continue

    def disable_audio(self):
        self.should_send_audio.clear()
        self.is_recording = False
        print("audio disabled")

    def enable_audio(self):
        self.should_send_audio.set()
        self.is_recording = True
        print("audio enabled")

    def reset_audio_enabled(self, delay=15):
        t = Timer(delay, self.disable_audio)
        t.start()

    async def send_wake_word(self) -> None:
        model = Model()

        try:
            async for audio_block in audio_input_generator(samplerate=16000):
                audio = np.frombuffer(audio_block, dtype=np.int16)
                model.predict(audio)

                for mdl in model.prediction_buffer.keys():
                    if mdl == "hey_jarvis":
                        scores = list(model.prediction_buffer[mdl])
                        if scores[-1] > 0.5 and not self.is_recording:
                            self.enable_audio()
                            self.reset_audio_enabled()

        except KeyboardInterrupt:
            pass

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        sent_audio = False
        try:
            async for audio_block in audio_input_generator():
                await self.should_send_audio.wait()
                connection = await self._get_connection()
                if not sent_audio:
                    asyncio.create_task(connection.send({"type": "response.cancel"}))
                    sent_audio = True

                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(cast(Any, audio_block)).decode("utf-8")
                )

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    app = RealtimeApp()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("done")
        pass
