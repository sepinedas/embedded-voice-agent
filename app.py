from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import RichLog, Static
from typing_extensions import override
from dotenv import load_dotenv

from src.audio import AudioStream, PlayAudio
from src.langchain_voice_agent import OpenAIVoiceReactAgent


load_dotenv()


class SessionStatus(Static):
    session_status = reactive("")

    @override
    def render(self) -> str:
        return "Connected." if self.session_status == "connected" else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)"
            if self.is_recording
            else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class RealtimeApp(App):
    CSS_PATH = "app.tcss"

    def __init__(self) -> None:
        super().__init__()

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionStatus(id="session-status")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        self.audio_stream = AudioStream()
        self.run_worker(self.connect())

    async def connect(self) -> None:
        agent = OpenAIVoiceReactAgent()

        # playAudio = PlayAudio()
        async def printTest(data):
            print(data)

        session_display = self.query_one(SessionStatus)
        session_display.session_status = "connected"

        await agent.aconnect(self.audio_stream.audio_generator(), printTest)

    async def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.exit()
            return

        status_indicator = self.query_one(AudioStatusIndicator)
        if event.key == "k":
            if status_indicator.is_recording:
                status_indicator.is_recording = False

                self.audio_stream.pause()
            else:
                status_indicator.is_recording = True
                self.audio_stream.resume()


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()
