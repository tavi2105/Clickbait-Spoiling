from interfaces.StrategyNLP import StrategyNLP
from models.ClickbaitSummaryType import ClickbaitSummaryType as cT


class SynopsisGenerator:
    def __init__(self, strategy_for_classification: StrategyNLP, strategy_for_phrase_summarization: StrategyNLP,
                 strategy_for_passage_summarization: StrategyNLP, strategy_for_multi_summarization: StrategyNLP):
        self.strategyForClassification = strategy_for_classification
        self.strategiesForSummarization = {cT.PASSAGE.name: strategy_for_passage_summarization,
                                           cT.PHRASE.name: strategy_for_phrase_summarization,
                                           cT.MULTI.name: strategy_for_multi_summarization}

    def classify(self, clickbait):
        return self.strategyForClassification.apply_on_single_clickbait(clickbait)

    def generate_synopsis(self, clickbait):
        click_type = self.classify(clickbait)
        return self.strategiesForSummarization[click_type].apply_on_single_clickbait(clickbait)


