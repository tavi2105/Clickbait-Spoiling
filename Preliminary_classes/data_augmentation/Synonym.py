import json
import nlpaug.augmenter.word as naw

import pandas as pd

TOPK = 20  # default=100
ACT = 'insert'  # "substitute"


class Synonym:
    def __init__(self):
        self.TOPK = 20
        self.ACT = 'insert'
        self.aug = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=10,
                                  aug_p=0.3, lang='eng',
                                  stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None,
                                  force_reload=False,
                                  verbose=0)

    def augmentation(self, json_list):
        new_entries = []
        counter = 1
        for entry in json_list:
            entry["targetParagraphs"] = self.paragraphs_augmentation(entry["targetParagraphs"])
            new_entries.append(entry)
            print("still working: " + str(counter))
            counter = counter + 1

        return new_entries

    def paragraphs_augmentation(self, paragraphs):
        paragraphs_augmented = []
        for paragraph in paragraphs:
            paragraphs_augmented.append(self.aug.augment(paragraph)[0])

        return paragraphs_augmented


aug = Synonym()

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
