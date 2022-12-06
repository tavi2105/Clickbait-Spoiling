from SynopsisGenerator import SynopsisGenerator
from ModelSelector import ModelSelector
from interfaces.Observer import Observer
from DataExtractor import DataExtractor
from models.Clickbait import Clickbait
from classification_algoritms.LogisticRegression import LogisticRegression
from classification_algoritms.NaiveBayes import NaiveBayes
from classification_algoritms.SVM import SVM
from evaluation_strategies.eval_for_classification import ClassificationEvaluation
from evaluation_strategies.eval_for_summarization import EvalForSummarization
from summarization_strategies.qa import QA


class AI(Observer):
    """
    This class is responsible for assembling and managing the tool for generating clickbait synopsis
    """

    def __init__(self):
        self.synopsis_generator: SynopsisGenerator = None
        self.modelSelectorForClassification: ModelSelector = ModelSelector([NaiveBayes(), LogisticRegression(), SVM()],
                                                                           ClassificationEvaluation())
        self.modelSelectorForSummarization: ModelSelector = ModelSelector([QA()],EvalForSummarization())
        self.dataExtractor: DataExtractor = DataExtractor()

    def load_necessary_data(self):
        pass

    def assemble_synopsis_generator(self):
        strategy_for_classification = self.modelSelectorForClassification.select_method(self.dataExtractor.get_train_data(),
                                                                                        self.dataExtractor.get_val_data())
        strategy_for_summarization = self.modelSelectorForSummarization.select_method(self.dataExtractor.get_train_data(),
                                                                                        self.dataExtractor.get_val_data())
        self.synopsis_generator = SynopsisGenerator(strategy_for_classification, strategy_for_summarization, strategy_for_summarization, strategy_for_summarization)

    def get_synopsis_generator(self) -> SynopsisGenerator:
        return self.synopsis_generator

    def generate_clickbait_synopsis(self, clickbait: Clickbait):
        if self.synopsis_generator is None:
            self.assemble_synopsis_generator()
        print("Spoiler Type: ", self.synopsis_generator.classify(clickbait))
        print("Spoiler synopsis: ")
        return self.synopsis_generator.generate_synopsis(clickbait)

    def update(self) -> None:
        # when updateNotifier notify that enough new data were added to database
        # we will retake the selection and training process
        pass
