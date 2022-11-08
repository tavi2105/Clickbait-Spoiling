import SynopsisGenerator
from Preliminary_classes.ModelSelector import ModelSelector
from interfaces.Observer import Observer


class AI(Observer):
    """
    This class is responsible for assembling and managing the tool for generating clickbait synopsis
    """
    def __init__(self):
        self.synopsis_generator: SynopsisGenerator = None
        self.modelSelectorForClassification: ModelSelector
        self.modelSelectorForSummarization: ModelSelector

    def load_necessary_data_from_database(self):
        pass

    def assemble_synopsis_generator(self):
        pass

    def get_synopsis_generator(self) -> SynopsisGenerator:
        pass

    def update(self) -> None:
        # when updateNotifier notify that enough new data were added to database
        # we will retake the selection and training process
        pass

