import logging
import time

import aspectlib, socket, sys
import aspectlib.debug

from Preliminary_classes.DataExtractor import DataExtractor
from Preliminary_classes.ModelSelector import ModelSelector
from Preliminary_classes.classification_algoritms.NaiveBayes import NaiveBayes
from Preliminary_classes.evaluation_strategies.eval_for_classification import ClassificationEvaluation
from tests.test_utils import TRAINING_DATA, VALIDATING_DATA


def time_log():
    @aspectlib.Aspect
    def time_log_aspect(*args):
        st = time.time()
        # logging.info("Execution started...")


        yield aspectlib.Proceed
        et = time.time()
        elapsed_time = et - st
        logging.info("Execution finished. Time: "+str(elapsed_time) )

    return time_log_aspect


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
with aspectlib.weave(
        ModelSelector,
        [time_log(),aspectlib.debug.log(
            module=False,
            use_logging="INFO",
         print_to=sys.stdout,
         stacktrace=None,
    )],
):
    m = ModelSelector([NaiveBayes()],ClassificationEvaluation())
    model = m.select_method(TRAINING_DATA,VALIDATING_DATA)
    print(model)


