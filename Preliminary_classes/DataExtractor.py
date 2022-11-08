from Preliminary_classes.interfaces.SingletonMeta import SingletonMeta


class DataExtractor(metaclass=SingletonMeta):
    def __init__(self):
        with open('data/train.jsonl', 'r', encoding='utf-8') as json_file:
            self.__data = list(json_file)
        # with open('data/augmented_contextual.jsonl', 'r', encoding='utf-8') as json_file:
        #     self.__data = self.__data + list(json_file)
        # with open('data/augmented_synonyms.jsonl', 'r', encoding='utf-8') as json_file:
        #     self.__data = self.__data + list(json_file)

    def get_data(self):
        return self.__data
