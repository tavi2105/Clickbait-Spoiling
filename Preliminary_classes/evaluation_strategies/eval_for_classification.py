from interfaces.StrategyForEvaluatingModels import StrategyForEvaluatingModels
from sklearn.metrics import accuracy_score


class ClassificationEvaluation(StrategyForEvaluatingModels):
    def __init__(self):
        self.correct_output = []

    def prepare_validating_data(self, validating_data):
        t =[]
        for i in validating_data:
            t.append(i.summary_type.name)
        self.correct_output = t

    def calculate_score(self, model,validating_data):
        return accuracy_score(self.correct_output,model.apply_on_list_of_clickbaits(validating_data))
