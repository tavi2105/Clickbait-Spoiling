import pandas
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression as LRmodel
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize

from interfaces.StrategyForClassification import StrategyForClassification
from interfaces.StrategyNLP import StrategyNLP
from models.Clickbait import Clickbait
from models.ClickbaitSolved import ClickbaitSolved
from models.ClickbaitSummaryType import ClickbaitSummaryType


class LogisticRegression(StrategyForClassification):
    def train(self, data):
        proc_data = self.prepare_data(data)
        # definim modelul
        preprocessor = self.assemble_classification_prep_pipeline(False,ngram_range=(1, 3), min_df=2, tokenizer=word_tokenize)

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LRmodel())
        ])
        self.model.fit(proc_data, proc_data["type"])
