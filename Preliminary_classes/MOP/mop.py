from pythonrv import rv
from MOP import validation
from DataExtractor import DataExtractor

data = DataExtractor()
train = data.get_train_data_dict()
val = data.get_val_data_dict()
train_clickbait = data.get_train_data()
val_clickbait = data.get_val_data()


@rv.monitor(clean=validation.clean_data, validate=validation.validate_data)
def spec_try_cleaning(event):
    if event.called_function == event.fn.validate:
        if len([old_ev for old_ev in event.history if old_ev.called_function == old_ev.fn.clean]) == 0:
            validation.clean_data(train)
        assert len([old_ev for old_ev in event.history if old_ev.called_function == old_ev.fn.clean]) > 0


@rv.monitor(up=validation.validate_data)
@rv.spec(when=rv.POST)
def check_validate(event):
    assert type(event.fn.up.result) is bool
    print("Result of validation is bool: OK")


@rv.monitor(up=validation.clean_data)
@rv.spec(when=rv.POST)
def check_clear(event):
    assert type(event.fn.up.result) is list
    print("Result of cleaning is list: OK")


rv.configure(error_handler=rv.LoggingErrorHandler())


@rv.monitor(f=validation.clean_data)
@rv.spec(level=rv.DEBUG)
def spec(event):
    pass


@rv.monitor(f=validation.clean_data)
@rv.spec(when=rv.PRE)
def spec_before(event):
    print("Before data cleaning: " + str(len(train)))


@rv.monitor(f=validation.clean_data)
@rv.spec(when=rv.POST)
def spec_after(event):
    print("After data cleaning: " + str(len(train)))


if __name__ == '__main__':
    validation.validate_data(train)
