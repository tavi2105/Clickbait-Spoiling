import operator

from interfaces import StrategyNLP, StrategyForEvaluatingModels


class ModelSelector:

    def __init__(self, models_to_evaluate: StrategyNLP, strategy_for_evaluating: StrategyForEvaluatingModels):
        self.models_to_evaluate = models_to_evaluate
        self.strategy_for_evaluating = strategy_for_evaluating
        pass

    def select_method(self, training_data, validating_data) -> StrategyNLP:
        results = {}
        cor = self.prepare_val_data(validating_data)
        for s in self.models_to_evaluate:
            s.train(training_data)
            results[s] = self.strategy_for_evaluating.calculate_score(s.apply_on_list_of_clickbaits(validating_data),cor)
        return max(results.items(), key=operator.itemgetter(1))[0]

    def prepare_val_data(self,val_data):
        t =[]
        for i in val_data:
            t.append(i.summary_type.name)
        return t
    def set_evaluating_strategy(self, strategy_for_evaluating: StrategyForEvaluatingModels):
        self.strategy_for_evaluating = strategy_for_evaluating
