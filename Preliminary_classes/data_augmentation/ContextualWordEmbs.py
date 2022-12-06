import json
import nlpaug.augmenter.word as naw

import pandas as pd

TOPK = 20  # default=100
ACT = 'insert'  # "substitute"


class ContextualWordEmbs:
    def __init__(self):
        self.TOPK = 20
        self.ACT = 'insert'
        self.aug_bert = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action=ACT, top_k=TOPK)

    def augmentation(self, json_list):
        counter = 1
        new_entries = []
        for entry in json_list:
            entry["targetParagraphs"] = self.paragraphs_augmentation(entry["targetParagraphs"])
            new_entries.append(entry)
            print("still working: " + str(counter))
            counter = counter + 1

        return new_entries

    def paragraphs_augmentation(self, paragraphs):
        paragraphs_augmented = []
        for paragraph in paragraphs:
            paragraphs_augmented.append(self.aug_bert.augment(paragraph))

        return paragraphs_augmented


aug = ContextualWordEmbs()

df_train = pd.read_json("../data/train.jsonl", lines=True)
result = df_train.to_json(orient="records")
parsed = json.loads(result)
df = pd.DataFrame(aug.augmentation(parsed))

df.to_json('../data/augmented_synonyms_train.jsonl', orient='records', lines=True)


df_train = pd.read_json("../data/validation.jsonl", lines=True)
result = df_train.to_json(orient="records")
parsed = json.loads(result)
df = pd.DataFrame(aug.augmentation(parsed))

df.to_json('../data/augmented_synonyms_val.jsonl', orient='records', lines=True)
