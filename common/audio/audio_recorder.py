import sounddevice as sd
import asyncio
from common.audio import CHANNELS, SAMPLE_RATE, CHUNK


async def audio_input_generator(
    channels=CHANNELS, samplerate=SAMPLE_RATE, chunk=CHUNK, dtype="int16"
):
    try:
        with sd.InputStream(
            samplerate=samplerate, channels=channels, dtype=dtype
        ) as stream:
            while True:
                data, overflowed = stream.read(CHUNK)
                if overflowed:
                    print(f"Audio input overflowed: {overflowed} blocks dropped!")
                yield data
                await asyncio.sleep(0)  # Allow other tasks to run

    except Exception as e:
        print(f"Error in audio input stream: {e}")
