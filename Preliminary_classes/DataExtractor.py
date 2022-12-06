import pandas as pd

from interfaces.SingletonMeta import SingletonMeta
from models.ClickbaitSolved import ClickbaitSolved


class DataExtractor(metaclass=SingletonMeta):
    def __init__(self):
        df_train = pd.read_json("./data/train.jsonl", lines=True)
        df_val = pd.read_json("./data/validation.jsonl", lines=True)
        # df_aug_con = pd.read_json("../data/augmented_contextual.jsonl", lines=True)
        # df_aug_syn = pd.read_json("../data/augmented_synonyms.jsonl", lines=True)

        self.__train_data_dict = df_train.to_dict(orient='records')
        self.__val_data_dict = df_val.to_dict(orient='records')
        # self.__aug_con_data_dict = df_aug_con.to_dict(orient='records')
        # self.__aug_syn_data_dict = df_aug_syn.to_dict(orient='records')

        self.__train_data = [ClickbaitSolved(**kwargs) for kwargs in df_train.to_dict(orient='records')]
        self.__val_data = [ClickbaitSolved(**kwargs) for kwargs in df_val.to_dict(orient='records')]
        # self.__aug_con_data = [ClickbaitSolved(**kwargs) for kwargs in df_aug_con.to_dict(orient='records')]
        # self.__aug_syn_data = [ClickbaitSolved(**kwargs) for kwargs in df_aug_syn.to_dict(orient='records')]

    def get_train_data(self):
        return self.__train_data

    def get_val_data(self):
        return self.__val_data

    # def get_aug_con_data(self):
    #     return self.__aug_con_data
    #
    # def get_aug_syn_data(self):
    #     return self.__aug_syn_data

    def get_train_data_dict(self):
        return self.__train_data_dict

    def get_val_data_dict(self):
        return self.__val_data_dict

    # def get_aug_con_data_dict(self):
    #     return self.__aug_con_data_dict
    #
    # def get_aug_syn_data_dict(self):
    #     return self.__aug_syn_data_dict

