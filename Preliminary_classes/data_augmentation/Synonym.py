import json
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nltk

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
        # self.file = open("augmented_synonyms.jsonl", "a", encoding='utf-8')
        # #with open('train.jsonl', 'r', encoding='utf-8') as json_file:
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
            paragraphs_augmented.append(self.aug.augment(paragraph))

        return paragraphs_augmented
