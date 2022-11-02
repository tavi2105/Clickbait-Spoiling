import json
import unittest

from Preliminary_classes.data_augmentation.ContextualWordEmbs import ContextualWordEmbs
from Preliminary_classes.data_augmentation.Synonym import Synonym


class TestingDataAugmentation(unittest.TestCase):

    def setUp(self):
        with open('data.jsonl', 'r', encoding='utf-8') as json_file:
            self.json_list = list(json_file)

        self.syno = Synonym()
        self.cont = ContextualWordEmbs()

    def test_synonym_augmentation_data_paragraphs_number(self):
        for json_str in self.json_list:
            entry = json.loads(json_str)
            test_paragraphs = self.syno.paragraphs_augmentation(entry["targetParagraphs"])
            self.assertEqual(len(test_paragraphs), len(entry["targetParagraphs"]))

    def test_synonym_augmentation_data_entries_number(self):
        test_entries = self.syno.augmentation(self.json_list)
        self.assertEqual(len(test_entries), len(self.json_list))

    def test_contextual_augmentation_data_paragraphs_number(self):
        for json_str in self.json_list:
            entry = json.loads(json_str)
            test_paragraphs = self.cont.paragraphs_augmentation(entry["targetParagraphs"])
            self.assertEqual(len(test_paragraphs), len(entry["targetParagraphs"]))

    def test_contextual_augmentation_data_entries_number(self):
        test_entries = self.cont.augmentation(self.json_list)
        self.assertEqual(len(test_entries), len(self.json_list))


if __name__ == '__main__':
    unittest.main()
