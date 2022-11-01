from abc import ABC, abstractmethod


class StrategyNLP(ABC):
    """
    The StrategyNLP interface declares operations common to all supported algorithms used for NLP.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """
    def __init__(self):
        self.model = None

    @abstractmethod
    def prepare_data(self, data):
        """Each algorithm has its own predefined preprocessing functions for preparing the data"""
        pass

    @abstractmethod
    def train(self, data):
        """Train the model on some dataset"""
        pass

    @abstractmethod
    def apply_on_single_clickbait(self, clickbait):
        """Apply the trained model on clickbait and return result"""
        pass

    @abstractmethod
    def apply_on_list_of_clickbaits(self, clickbait_list):
        """Apply the trained model on clickbait and return result"""
        pass

    @abstractmethod
    def save(self):
        """Store the model to persistent storage for reuse"""
        pass

    @abstractmethod
    def load(self):
        """Load the model from persistent storage"""
        pass
