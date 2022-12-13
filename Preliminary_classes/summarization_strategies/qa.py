import pandas
from transformers import pipeline

from interfaces.StrategyNLP import StrategyNLP
from models.Clickbait import Clickbait
from models.ClickbaitSolved import ClickbaitSolved


class QA(StrategyNLP):
    def prepare_data(self, data: [Clickbait]):
        if isinstance(data[0], ClickbaitSolved):
            return pandas.DataFrame.from_records([{
                "context": "\n".join(s.targetParagraphs),
                "question": "\n".join(s.postText),
                "spoiler": s.spoiler,
            }
                for s in data])
        return pandas.DataFrame.from_records([{
            "context": "\n".join(s.targetParagraphs),
            "question": "\n".join(s.postText),
        }
            for s in data])

    def train(self, data):
        model_checkpoint = "huggingface-course/bert-finetuned-squad"
        self.model = pipeline("question-answering", model=model_checkpoint)

    def apply_on_single_clickbait(self, clickbait):
        preproc_data = self.prepare_data([clickbait])
        return self.model(preproc_data["question"].tolist()[0], preproc_data["context"].tolist()[0])["answer"]

    def apply_on_list_of_clickbaits(self, clickbait_list):
        sol = []
        for c in clickbait_list:
            sol.append(self.apply_on_single_clickbait(c))
        return sol

    # Aceste metode vor fi implementate cand vom stabili o modalitate de a stoca modelele
    # dar momentan nu se încadrează in prioritatea acestor prime 2 iteratii
    def save(self):
        pass

    def load(self):
        pass
