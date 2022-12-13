from abc import ABC, abstractmethod

import pandas
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from interfaces.StrategyNLP import StrategyNLP
from models.Clickbait import Clickbait
from models.ClickbaitSolved import ClickbaitSolved
from models.ClickbaitSummaryType import ClickbaitSummaryType


class StrategyForClassification(StrategyNLP, ABC):
    """
    The StrategyNLP interface declares operations common to all supported algorithms used for NLP.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

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

    def assemble_classification_prep_pipeline(self, sublinear_tf: bool, **kwargs):
        title_pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words="english")),
            ('tdf', TfidfTransformer(sublinear_tf=sublinear_tf))
        ])
        description_pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words="english")),
            ('tdf', TfidfTransformer(sublinear_tf=sublinear_tf))
        ])
        par_pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words="english", **kwargs)),
            ('tdf', TfidfTransformer(sublinear_tf=sublinear_tf))
        ])
        # definim modelul
        return ColumnTransformer([
            ('targetTitle', title_pipeline, 'targetTitle'),
            ('postText', description_pipeline, 'postText'),
            ('targetParagraphs', par_pipeline, 'targetParagraphs'),
        ])

    def apply_on_single_clickbait(self, clickbait):
        preproc_data = self.prepare_data([clickbait])
        return ClickbaitSummaryType(self.model.predict(preproc_data)[0]).name

    def apply_on_list_of_clickbaits(self, clickbait_list):
        preproc_data = self.prepare_data(clickbait_list)
        sol = self.model.predict(preproc_data)
        return [ClickbaitSummaryType(s).name for s in sol]

    # Aceste metode vor fi implementate cand vom stabili o modalitate de a stoca modelele
    # dar momentan nu se încadrează in prioritatea acestor prime 2 iteratii
    def save(self):
        pass

    def load(self):
        pass
