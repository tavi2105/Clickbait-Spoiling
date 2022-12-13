from interfaces.StrategyForEvaluatingModels import StrategyForEvaluatingModels
from sentence_transformers import SentenceTransformer, util


class EvalForSummarization(StrategyForEvaluatingModels):
    def __init__(self):
        self.correct_output = []
        self.model = SentenceTransformer('stsb-roberta-large')

    def prepare_validating_data(self, validating_data):
        solutions = []
        for i in validating_data:
            solutions.append(i.spoiler[0])
        self.correct_output = solutions

    def convert_to_embedding(self, data: [str]):
        return self.model.encode(data[:5], convert_to_tensor=True)

    def calculate_score(self, model, validating_data):
        correct_output_embedding = self.convert_to_embedding(self.correct_output)
        actual_output_embedding = self.convert_to_embedding(
                                                    [x for x in model.apply_on_list_of_clickbaits(validating_data[:5])]
                                                   )
        scores = []
        for i in range(len(actual_output_embedding)):
            scores.append(util.pytorch_cos_sim(correct_output_embedding[i], actual_output_embedding[i]).item())

        score = sum(scores) / len(actual_output_embedding)
        return score
