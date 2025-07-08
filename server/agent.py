from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Coroutine
from pydantic import BaseModel, Field, SecretStr
from langchain_core.utils import secret_from_env
from langchain_core._api import beta
import json
import websockets
import asyncio

from utils import amerge

DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-10-01"
DEFAULT_URL = "wss://api.openai.com/v1/realtime"


@asynccontextmanager
async def connect(*, api_key: str, model: str, url: str) -> AsyncGenerator[
    tuple[
        Callable[[dict[str, Any] | str], Coroutine[Any, Any, None]],
        AsyncIterator[dict[str, Any]],
    ],
    None,
]:

    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    url = url or DEFAULT_URL
    url += f"?model={model}"

    websocket = await websockets.connect(url, extra_headers=headers)

    try:

        async def send_event(event: dict[str, Any] | str) -> None:
            formatted_event = json.dumps(
                event) if isinstance(event, dict) else event
            await websocket.send(formatted_event)

        async def event_stream() -> AsyncIterator[dict[str, Any]]:
            async for raw_event in websocket:
                yield json.loads(raw_event)

        stream: AsyncIterator[dict[str, Any]] = event_stream()

        yield send_event, stream
    finally:
        await websocket.close()


class OpenAIVoiceAgent(BaseModel):
    model: str = DEFAULT_MODEL
    api_key: SecretStr = Field(
        alias="openai_api_key",
        default_factory=secret_from_env("OPENAI_API_KEY", default=""),
    )
    url: str = Field(default=DEFAULT_URL)

    async def aconnect(self, input_stream: AsyncIterator[str], send_output_chunk: Callable[[str], Coroutine[Any, Any, None]]) -> None:
        async with connect(
            model=self.model, api_key=self.api_key.get_secret_value(), url=self.url
        ) as (
            model_send,
            model_receive_stream,
        ):
            await model_send(
                {
                    "type": "session.update",
                    "session": {
                        "input_audio_transcription": {
                            "model": "whisper-1",
                        },
                    },
                }
            )

            async for stream_key, data_row in amerge(input_mic=input_stream, output_speaker=model_receive_stream):
                try:
                    data = json.loads(data_row) if isinstance(data_row, str) else data_row
                except json.JSONDecodeError:
                    data = {"error": "Invalid JSON received from model."}
                continue

                if stream_key == "input_mic":
                    await model_send(data)
                elif stream_key == "output_speaker":
                    t = data["type"]

                    if t == "response.audio.delta":
                        await send_output_chunk(json.dumps(data))
                    elif t == "input_audio_buffer.speech_started":
                        print("interrupt")
                        send_output_chunk(json.dumps(data))
                    elif t == "error":
                        print("error:", data)
                    else:
                        print(t)
