from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
import pandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from interfaces.StrategyForClassification import StrategyForClassification
from models.Clickbait import Clickbait
from models.ClickbaitSolved import ClickbaitSolved
from models.ClickbaitSummaryType import ClickbaitSummaryType
from interfaces.StrategyNLP import StrategyNLP
from nltk.tokenize import NLTKWordTokenizer


class SVM(StrategyForClassification):

    def train(self, data):
        proc_data = self.prepare_data(data)
        # definim modelul
        preprocessor = self.assemble_classification_prep_pipeline(True, ngram_range=(1, 3), min_df=2
                                                                  , tokenizer=NLTKWordTokenizer().tokenize)

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', SVC(C=10, gamma=0.1))
        ])
        self.model.fit(proc_data, proc_data["type"])


