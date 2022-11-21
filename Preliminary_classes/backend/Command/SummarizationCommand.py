from .CommandInterface import Command
from models.Clickbait import Clickbait
from FacadeAI import FacadeAI


class Summarization(Command):
    # summarization command is for requesting synopsis from the AI

    def __init__(self, clickbait: Clickbait) -> None:
        self._receiver = FacadeAI()
        self._clickbait = clickbait

    def execute(self) -> None:
        self._receiver.apply_synopsys(self._clickbait)
