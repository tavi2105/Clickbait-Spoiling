from interfaces import SingletonMeta


class DataExtractor(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.__file = "file.jsom"
        self.__data = []

    def read_data(self):
        # read data from file
        pass

    def write_data(self):
        # write data in file
        pass

    def get_data(self):
        return self.__data
