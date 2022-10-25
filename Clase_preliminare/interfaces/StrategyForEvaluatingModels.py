from abc import ABC, abstractmethod


class StrategyForEvaluatingModels(ABC):
    """
    This interface declares operations common to all supported algorithms used for evaluating used models.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def calculate_score(self, model_output, correct_output):
        pass
