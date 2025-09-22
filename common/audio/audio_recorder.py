import asyncio
from common.audio import CHANNELS, CHUNK_LENGTH_S, SAMPLE_RATE
import sounddevice as sd


async def audio_input_generator(
    channels=CHANNELS, samplerate=SAMPLE_RATE, chunk_length_s=CHUNK_LENGTH_S
):
    read_size = int(chunk_length_s * samplerate)
    stream = sd.InputStream(channels=channels, samplerate=samplerate, dtype="int16")
    stream.start()

    try:
        while True:
            if stream.read_available < read_size:
                await asyncio.sleep(0)
                continue

            data, _ = stream.read(read_size)
            yield data
            await asyncio.sleep(0)

    except Exception as e:
        print(f"Error in audio input stream: {e}")
    finally:
        stream.stop()
        stream.close()
