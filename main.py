import asyncio
import base64
from threading import Timer
from typing import Any, Optional, cast
import sounddevice as sd

from gpiozero import LED, InputDevice

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from common.audio.audio_player import AudioPlayerAsync

from common.audio.audio_recorder import audio_input_generator

load_dotenv()
connected_led = LED(27)
wake = LED(17)
input_pin = InputDevice(15, pull_up=False)

DEFAULT_MODEL = "gpt-realtime"
VOICE = "coral"


class RealtimeApp:
    client: AsyncOpenAI
    connection: AsyncRealtimeConnection | None
    connected: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    is_awake: bool
    is_receiving: bool
    activity_timer: Optional[Timer]
    lock: asyncio.Lock

    def __init__(self) -> None:
        self.client = AsyncOpenAI()
        self.connected = asyncio.Event()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.is_awake = False
        self.is_receiving = False
        self.activity_timer = None
        self.lock = asyncio.Lock()

    async def run(self):
        print("running")
        print(sd.query_devices())

        await self.awake_mic()
        await asyncio.gather(
            self.handle_realtime_connection(),
            self.send_mic_audio(),
            self.handle_button(),
        )

    async def handle_button(self):
        while True:
            if input_pin.is_active and not self.is_receiving:
                await self.awake_mic()
            else:
                await self.sleep_mic()
            await asyncio.sleep(0)

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
                if event.type == "response.output_audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)

                    if self.is_awake:
                        await self.sleep_mic()

                    async with self.lock:
                        self.is_receiving = True
                    continue

                if event.type == "response.done":
                    asyncio.create_task(self.awake_mic_after_response_done())
                    continue

                if event.type == "response.output_audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta
                    continue

    async def awake_mic_after_response_done(self):
        async with self.lock:
            while self.audio_player.queue:
                await asyncio.sleep(0)
            self.is_receiving = False
        await asyncio.sleep(100)
        await self.awake_mic()

    async def sleep_mic(self):
        async with self.lock:
            self.is_awake = False
            wake.off()

    async def awake_mic(self):
        async with self.lock:
            self.is_awake = True
            wake.on()

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        sent_audio = False

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


if __name__ == "__main__":
    app = RealtimeApp()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("done")
        pass
