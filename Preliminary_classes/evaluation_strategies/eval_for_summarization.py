from interfaces.StrategyForEvaluatingModels import StrategyForEvaluatingModels
from sentence_transformers import SentenceTransformer, util


class EvalForSummarization(StrategyForEvaluatingModels):
    def __init__(self):
        self.correct_output = []
        self.model = SentenceTransformer('stsb-roberta-large')

    def prepare_validating_data(self, validating_data):
        t = []
        for i in validating_data:
            t.append(i.spoiler[0])
        self.correct_output = t

    def convert_to_embedding(self, data: [str]):
        print(data)
        return self.model.encode(data[:5], convert_to_tensor=True)

    def calculate_score(self, model,validating_data):
        em1 = self.convert_to_embedding(self.correct_output)
        em2 = self.convert_to_embedding([x["answer"] for x in model.apply_on_list_of_clickbaits(validating_data[:5])])
        scores = []
        for i in range(len(em2)):
            scores.append(util.pytorch_cos_sim(em1[i], em2[i]).item())

        score =sum(scores)/ len(em2)
        print("Eval for summarization:",score)
        return score
