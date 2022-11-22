from pandas.io import json
from pythonrv import rv
from MOP import validation

from DataExtractor import DataExtractor

# data = DataExtractor()
with open('../data/train.jsonl', 'r', encoding='utf-8') as json_file:
    data = [json.loads(line) for line in json_file]


@rv.monitor(clean=validation.clean_data, validate=validation.validate_data)
def spec_try_cleaning(event):
    if event.called_function == event.fn.validate:
        if len([old_ev for old_ev in event.history if old_ev.called_function == old_ev.fn.clean]) == 0:
            validation.clean_data(data)
        assert len([old_ev for old_ev in event.history if old_ev.called_function == old_ev.fn.clean]) > 0


if __name__ == '__main__':
    print("Before data validation: " + str(len(data)))
    validation.validate_data(data)
    print("After data validation: " + str(len(data)))
    # validation.validate_data(data.get_train_data())
