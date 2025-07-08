from starlette.routing import Route
from starlette.websockets import WebSocket
import uvicorn
from starlette.applications import Starlette

from agent import OpenAIVoiceAgent
from utils import websocket_stream


async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    browser_receive_stream = websocket_stream(websocket)

    agent = OpenAIVoiceAgent()

    await agent.aconnect(
        browser_receive_stream, websocket.send_text,
    )

routes = [Route('/ws', websocket_endpoint)]

app = Starlette(debug=True, routes=routes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
