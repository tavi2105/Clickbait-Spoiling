import logging
import os
from datetime import datetime
import time
from openpyxl import load_workbook
import aspectlib, sys
import aspectlib.debug
from Preliminary_classes.ModelSelector import ModelSelector
from Preliminary_classes.classification_algoritms.LogisticRegression import LogisticRegression
from Preliminary_classes.classification_algoritms.NaiveBayes import NaiveBayes
from Preliminary_classes.evaluation_strategies.eval_for_classification import ClassificationEvaluation
from tests.test_utils import TRAINING_DATA, VALIDATING_DATA


def time_log():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    @aspectlib.Aspect(bind=True)
    def time_log_aspect(cutpoint, *args):
        st = time.time()
        logging.info("Execution of " + cutpoint.__qualname__ + " started...")

        result = yield aspectlib.Proceed
        result_log = result
        if hasattr(result,'size') and result.size>3:
            result_log = "Too large to display"
        logging.info("Result => " + str(result_log))
        if hasattr(result, "__class__"):
            logging.info("Result type => " + result.__class__.__name__)
        et = time.time()
        elapsed_time = et - st
        logging.info(
            "Execution of " + cutpoint.__qualname__ + " finished...\n          Time of execution: " + str(elapsed_time))

    return time_log_aspect


def xml_result_caching():
    @aspectlib.Aspect
    def xml_result_caching_aspect(*args):
        model_name = args[1].__class__.__name__
        result=yield aspectlib.Proceed
        wb = load_workbook("results_caching/results.xlsx")
        ws = wb[model_name]
        ws.append([datetime.now().strftime("%d/%m/%Y %H:%M:%S"),result])
        wb.save("results_caching/results.xlsx")
    return xml_result_caching_aspect


with aspectlib.weave(
        ModelSelector,
        [time_log(),
         # aspectlib.debug.log(
         #  module=True,
         # use_logging="INFO",
         # print_to=sys.stdout,
         # stacktrace=None,)
         ],
),aspectlib.weave(
        NaiveBayes,
        time_log()), \
  aspectlib.weave(ClassificationEvaluation.calculate_score, xml_result_caching()):
    m = ModelSelector([NaiveBayes(), LogisticRegression()], ClassificationEvaluation())
    model = m.select_method(TRAINING_DATA, VALIDATING_DATA)
