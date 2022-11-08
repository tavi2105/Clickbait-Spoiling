from Preliminary_classes.interfaces.StrategyForEvaluatingModels import StrategyForEvaluatingModels
from sklearn.metrics import accuracy_score


class ClassificationEvaluation(StrategyForEvaluatingModels):
    def calculate_score(self, model_output, correct_output):
        return accuracy_score(correct_output,model_output)
