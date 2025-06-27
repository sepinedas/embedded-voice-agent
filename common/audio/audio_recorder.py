import sounddevice as sd
import pyaudio
import asyncio
from common.audio import CHANNELS, SAMPLE_RATE, CHUNK


async def audio_input_generator(
    channels=CHANNELS, samplerate=SAMPLE_RATE, chunk=CHUNK, dtype="int16"
):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=samplerate, input=True,frames_per_buffer=chunk)

    try:
        while True:
            data = stream.read(chunk)
            yield data
            await asyncio.sleep(0)  # Allow other tasks to run

    except Exception as e:
        print(f"Error in audio input stream: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
