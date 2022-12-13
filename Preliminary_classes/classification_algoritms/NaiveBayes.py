import pandas
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize

from interfaces.StrategyForClassification import StrategyForClassification
from interfaces.StrategyNLP import StrategyNLP
from models.Clickbait import Clickbait
from models.ClickbaitSolved import ClickbaitSolved
from models.ClickbaitSummaryType import ClickbaitSummaryType


class NaiveBayes(StrategyForClassification):

    def train(self, data):
        proc_data = self.prepare_data(data)
        # definim modelul
        preprocessor = self.assemble_classification_prep_pipeline(True, ngram_range=(1, 3), min_df=2, max_df=0.7,
                                                                  tokenizer=word_tokenize)

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('clf',
             MultinomialNB())
        ])

        self.model.fit(proc_data, proc_data["type"])
