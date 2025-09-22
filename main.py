import asyncio
import base64
from threading import Timer
from typing import Any, cast
import numpy as np
import sounddevice as sd
from scipy.signal import resample

from gpiozero import LED

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openwakeword.model import Model
from common.audio.audio_player import AudioPlayerAsync

from common.audio.audio_recorder import audio_input_generator

load_dotenv()
connected_led = LED(27)
wake = LED(17)

DEFAULT_MODEL = "gpt-realtime"
VOICE = "coral"


class RealtimeApp:
    client: AsyncOpenAI
    connection: AsyncRealtimeConnection | None
    connected: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    is_awake: bool

    def __init__(self) -> None:
        self.client = AsyncOpenAI()
        self.connected = asyncio.Event()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.is_awake = False

    async def run(self):
        print("running")
        print(sd.query_devices())
        self.enable_audio()
        await asyncio.gather(
            self.handle_realtime_connection(),
            self.send_mic_audio(),
        )

    async def handle_realtime_connection(self) -> None:
        async with self.client.realtime.connect(model=DEFAULT_MODEL) as conn:
            self.connection = conn
            self.connected.set()
            connected_led.on()

            await conn.session.update(
                session={
                    "audio": {
                        "input": {"turn_detection": {"type": "server_vad"}},
                        "output": {"voice": VOICE},
                    },
                    "model": DEFAULT_MODEL,
                    "type": "realtime",
                }
            )

            acc_items: dict[str, Any] = {}  # noqa: F821

            async for event in conn:
                print(event.type)
                if event.type == "response.output_audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)

                    self.disable_audio()
                    continue

                if event.type == "response.done":
                    while self.audio_player.queue:
                        await asyncio.sleep(0)
                    self.reset_audio_disabled()
                    self.reset_audio_enabled()
                    continue

                if event.type == "response.output_audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta

                    continue

    def disable_audio(self):
        self.is_awake = False
        wake.off()
        print("audio disabled")

    def enable_audio(self):
        self.is_awake = True
        wake.on()
        print("audio enabled")

    def reset_audio_enabled(self, delay=30):
        t = Timer(delay, self.disable_audio)
        t.start()

    def reset_audio_disabled(self, delay=2):
        t = Timer(delay, self.enable_audio)
        t.start()

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        sent_audio = False
        model = Model()
        target_sample_rate = 16000
        original_sample_rate = 24000

        async for data in audio_input_generator():
            if self.is_awake:
                connection = await self._get_connection()
                if not sent_audio:
                    asyncio.create_task(connection.send({"type": "response.cancel"}))
                    sent_audio = True

                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(cast(Any, data)).decode("utf-8")
                )
                await asyncio.sleep(0)
            else:
                audio = np.frombuffer(data, dtype=np.int16)
                num_samples_original = len(audio)
                num_samples_downsampled = int(
                    num_samples_original * (target_sample_rate / original_sample_rate)
                )
                downsampled_audio = resample(audio, num_samples_downsampled)

                model.predict(downsampled_audio)

                for mdl in model.prediction_buffer.keys():
                    if mdl == "alexa":
                        scores = list(model.prediction_buffer[mdl])
                        if scores[-1] > 0.5:
                            self.enable_audio()
                            self.reset_audio_enabled()
                await asyncio.sleep(0)


if __name__ == "__main__":
    app = RealtimeApp()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("done")
        pass
