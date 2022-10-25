from .SummarizationCommand import Summarization
from .TrainingTriggerCommand import TrainingTrigger


class Invoker:
    # the invoker knows which action to execute depending on wich function is called

    def summarization(self, text):
        _summarization = Summarization(text)
        _summarization.execute()

    def feedbcak(self, text):
        _feedback = TrainingTrigger(text)
        _feedback.execute()
