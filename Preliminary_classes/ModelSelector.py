import operator

from interfaces import StrategyNLP, StrategyForEvaluatingModels


class ModelSelector:

    def __init__(self, models_to_evaluate: StrategyNLP, strategy_for_evaluating: StrategyForEvaluatingModels):
        self.models_to_evaluate = models_to_evaluate
        self.strategy_for_evaluating = strategy_for_evaluating
        pass

    def select_method(self, training_data, validating_data) -> StrategyNLP:
        results = {}
        self.strategy_for_evaluating.prepare_validating_data(validating_data)
        for model in self.models_to_evaluate:
            model.train(training_data)
            results[model] = self.strategy_for_evaluating.calculate_score(model, validating_data)
        return max(results.items(), key=operator.itemgetter(1))[0]

    def set_evaluating_strategy(self, strategy_for_evaluating: StrategyForEvaluatingModels):
        self.strategy_for_evaluating = strategy_for_evaluating
