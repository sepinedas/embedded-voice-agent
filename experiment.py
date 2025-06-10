from src.audio import Audio
import asyncio
from dotenv import load_dotenv

load_dotenv()


async def main():
    running = True
    audio = Audio()

    async def stop_recording():
        input("Press <any key> again to stop recording")
        audio.stop_recording()

    while running:
        text = input("Press <enter> to send a voice command, <q> to exit")

        if text == "":
            await asyncio.gather(
                asyncio.to_thread(audio.start_recording), stop_recording()
            )
            audio.initialize_audio()
            print("here")

        if text == "q":
            running = False


if __name__ == "__main__":
    asyncio.run(main())
