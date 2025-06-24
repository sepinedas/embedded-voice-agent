import sounddevice as sd
import asyncio
from common.audio import CHANNELS, SAMPLE_RATE, CHUNK_LENGTH_S


async def audio_input_generator():
    read_size = int(SAMPLE_RATE * 0.02)
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16"
        ) as stream:
            while True:
                data, overflowed = stream.read(read_size)
                if overflowed:
                    print(f"Audio input overflowed: {overflowed} blocks dropped!")
                yield data
                await asyncio.sleep(0)  # Allow other tasks to run

    except Exception as e:
        print(f"Error in audio input stream: {e}")
