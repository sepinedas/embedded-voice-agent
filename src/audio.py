import pyaudio
import asyncio
from typing import AsyncIterator

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 512
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "recordedFile.wav"


class AudioStream:
    def __init__(self, format=FORMAT, channels=CHANNELS, rate=RATE, chunk=CHUNK):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )
        self.paused = True  # Flag to control pause/resume

    async def audio_generator(self) -> AsyncIterator[str]:
        while True:
            if not self.paused:
                data = self.stream.read(self.chunk)
                yield data
            await asyncio.sleep(0.001)  # Small delay to avoid busy-waiting

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class PlayAudio:
    def __init__(self):
        p = pyaudio.PyAudio()
        self.stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)

    def play(self, data):
        self.stream(data)
