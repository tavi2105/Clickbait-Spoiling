from .CommandInterface import Command
import DataExtractor

class TrainingTrigger(Command):
   #the trigger command is for adding user versions for synopsis and trigger trainings on new data

    def __init__(self, receiver: DataExtractor, data, review) -> None:
        self._receiver = receiver
        self._data = data
        self._review = review

    def execute(self) -> None:
        self._receiver.write_data(self._data)