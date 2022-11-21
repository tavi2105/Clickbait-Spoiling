import pandas as pd

from interfaces.SingletonMeta import SingletonMeta
from models.ClickbaitSolved import ClickbaitSolved


class DataExtractor(metaclass=SingletonMeta):
    def __init__(self):
        with open('data/train.jsonl', 'r', encoding='utf-8') as json_file:
            self.__data = list(json_file)
        df_train = pd.read_json("data/train.jsonl", lines=True)
        df_val = pd.read_json("data/validation.jsonl", lines=True)
        print(df_train.to_dict(orient='records')[0].keys())
        self.__train_data = [ClickbaitSolved(**kwargs) for kwargs in df_train.to_dict(orient='records')]
        self.__val_data = [ClickbaitSolved(**kwargs) for kwargs in df_val.to_dict(orient='records')]

        # with open('data/augmented_contextual.jsonl', 'r', encoding='utf-8') as json_file:
        #     self.__data = self.__data + list(json_file)
        # with open('data/augmented_synonyms.jsonl', 'r', encoding='utf-8') as json_file:
        #     self.__data = self.__data + list(json_file)

    def get_data(self):
        return self.__data

    def get_train_data(self):
        return self.__train_data

    def get_val_data(self):
        return self.__val_data


