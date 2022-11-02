import pandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression as LRmodel
from sklearn.pipeline import Pipeline

from Preliminary_classes.interfaces.StrategyNLP import StrategyNLP
from Preliminary_classes.models.Clickbait import Clickbait
from Preliminary_classes.models.ClickbaitSolved import ClickbaitSolved
from Preliminary_classes.models.ClickbaitSummaryType import ClickbaitSummaryType


class LogisticRegression(StrategyNLP):
    def prepare_data(self, data: [Clickbait]):
        if isinstance(data[0], ClickbaitSolved):
            return pandas.DataFrame.from_records([{
                "targetParagraphs": "\n".join(s.targetParagraphs),
                "type": s.summary_type.value
            }
                for s in data])
        return pandas.DataFrame.from_records([{
            "targetParagraphs": "\n".join(s.targetParagraphs),
        }
            for s in data])

    def train(self, data):
        proc_data = self.prepare_data(data)
        self.model = Pipeline(
            steps=[
                (
                    # aici ii dam algoritmul ce se va ocupa cu prelucrarea textului, transformand-ul intr-un vector cu valori numerice
                    "count_verctorizer", CountVectorizer(stop_words='english')
                ),
                ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True, norm='l2', )),

                ('clf', LRmodel())
            ])
        self.model.fit(proc_data["targetParagraphs"], proc_data["type"])

    def apply_on_single_clickbait(self, clickbait):
        preproc_data = self.prepare_data([clickbait])
        return ClickbaitSummaryType(self.model.predict(preproc_data)[0]).name

    def apply_on_list_of_clickbaits(self, clickbait_list):
        preproc_data = self.prepare_data(clickbait_list)
        sol = self.model.predict(preproc_data['targetParagraphs'])
        return [ClickbaitSummaryType(s).name for s in sol]

    def save(self):
        pass

    def load(self):
        pass

