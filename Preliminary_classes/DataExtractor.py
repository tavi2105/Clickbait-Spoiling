import json

import pandas as pd

from interfaces.SingletonMeta import SingletonMeta
from models.ClickbaitSolved import ClickbaitSolved


class DataExtractor(metaclass=SingletonMeta):
    def __init__(self):
        df_train = pd.read_json("data/train.jsonl", lines=True)
        df_val = pd.read_json("data/validation.jsonl", lines=True)
        print(df_train.to_dict())
        self.__train_data = [ClickbaitSolved(**kwargs) for kwargs in df_train.to_dict(orient='records')]
        self.__val_data = [ClickbaitSolved(**kwargs) for kwargs in df_val.to_dict(orient='records')]

    def get_train_data(self):
        return self.__train_data

    def get_val_data(self):
        return self.__val_data

