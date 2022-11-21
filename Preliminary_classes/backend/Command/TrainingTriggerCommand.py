from .CommandInterface import Command
from FacadeAI import FacadeAI


class TrainingTrigger(Command):
    # the trigger command is for adding user versions for synopsis and trigger trainings on new data

    def __init__(self, data, review) -> None:
        self._receiver = FacadeAI()
        self._data = data
        self._review = review

    def execute(self) -> None:
        self._receiver.write_data(self._data)
