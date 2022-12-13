import pandas
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression as LRmodel
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from interfaces.StrategyNLP import StrategyNLP
from models.Clickbait import Clickbait
from models.ClickbaitSolved import ClickbaitSolved
from models.ClickbaitSummaryType import ClickbaitSummaryType


class LogisticRegression(StrategyNLP):
    def prepare_data(self, data: [Clickbait]):
        if isinstance(data[0], ClickbaitSolved):
            return pandas.DataFrame.from_records([{
                "targetParagraphs": "\n".join(s.targetParagraphs),
                "postText": "\n".join(s.postText),
                "targetTitle": s.targetTitle,
                "type": s.summary_type.value
            }
                for s in data])
        return pandas.DataFrame.from_records([{
            "targetParagraphs": "\n".join(s.targetParagraphs),
            "postText": "\n".join(s.postText),
            "targetTitle": s.targetTitle,
        }
            for s in data])

    def train(self, data):
        proc_data = self.prepare_data(data)
        title_pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words="english")),
            ('tdf', TfidfTransformer())
        ])
        description_pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words="english")),
            ('tdf', TfidfTransformer())
        ])
        par_pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=2, tokenizer=word_tokenize)),
            ('tdf', TfidfTransformer())
        ])
        # definim modelul
        preprocessor = ColumnTransformer([
            ('targetTitle', title_pipeline, 'targetTitle'),
            ('postText', description_pipeline, 'postText'),
            ('targetParagraphs', par_pipeline, 'targetParagraphs'),
        ])

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LRmodel())
        ])
        self.model.fit(proc_data, proc_data["type"])

    def apply_on_single_clickbait(self, clickbait):
        preproc_data = self.prepare_data([clickbait])
        return ClickbaitSummaryType(self.model.predict(preproc_data)[0]).name

    def apply_on_list_of_clickbaits(self, clickbait_list):
        preproc_data = self.prepare_data(clickbait_list)
        sol = self.model.predict(preproc_data)
        return [ClickbaitSummaryType(s).name for s in sol]

    def save(self):
        pass

    def load(self):
        pass
