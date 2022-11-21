import json
import nlpaug.augmenter.word as naw

TOPK = 20  # default=100
ACT = 'insert'  # "substitute"


class ContextualWordEmbs():
    def __init__(self):
        self.TOPK = 20
        self.ACT = 'insert'
        self.aug_bert = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action=ACT, top_k=TOPK)
        # self.file = open("augmented_contextual.jsonl", "a", encoding='utf-8')
        # with open('train.jsonl', 'r', encoding='utf-8') as json_file:
        # self.json_list = list(json_file)

    def augmentation(self, json_list):
        new_entries = []
        for json_str in json_list:
            entry = json.loads(json_str)
            entry["targetParagraphs"] = self.paragraphs_augmentation(entry["targetParagraphs"])
            new_entries.append(str(entry))

        return new_entries

    def paragraphs_augmentation(self, paragraphs):
        paragraphs_augmented = []
        for paragraph in paragraphs:
            paragraphs_augmented.append(self.aug_bert.augment(paragraph))

        return paragraphs_augmented
