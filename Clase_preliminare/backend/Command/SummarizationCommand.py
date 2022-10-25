from .CommandInterface import Command
from modele.Clickbait import Clickbait
from SynopsisGenerator import SynopsisGenerator


class Summarization(Command):
  #summarization command is for requesting synopsis from the AI

    def __init__(self, receiver: SynopsisGenerator, clickbait: Clickbait) -> None:

        self._receiver = receiver
        self._clickbait = clickbait

    def execute(self) -> None:
        self._receiver.generate_synopsis(self._clickbait)
