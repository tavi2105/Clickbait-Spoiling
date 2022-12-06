import unittest

from DataExtractor import DataExtractor


class TestData(unittest.TestCase):
    def setUp(self):
        self.app = DataExtractor()

    def test_train_data_count(self):
        self.assertEqual(len(self.app.get_train_data()), 3200)

    def test_val_data_count(self):
        self.assertEqual(len(self.app.get_val_data()), 800)

    def test_aug_con_train_data_count(self):
        self.assertEqual(len(self.app.get_aug_con_train_data()), len(self.app.get_train_data()))

    def test_aug_syn_train_data_count(self):
        self.assertEqual(len(self.app.get_aug_syn_train_data()), len(self.app.get_train_data()))

    def test_aug_con_val_data_count(self):
        self.assertEqual(len(self.app.get_aug_con_val_data()), len(self.app.get_val_data()))

    def test_aug_syn_val_data_count(self):
        self.assertEqual(len(self.app.get_aug_syn_val_data()), len(self.app.get_val_data()))

    def test_train_data_not_null(self):
        for entry in self.app.get_train_data_dict():
            self.assertNotEqual(entry['uuid'], None)
            self.assertNotEqual(entry['postText'], None)
            self.assertNotEqual(entry['targetParagraphs'], None)
            self.assertNotEqual(entry['targetTitle'], None)
            self.assertNotEqual(entry['spoiler'], None)
            self.assertNotEqual(entry['spoilerPositions'], None)
            self.assertNotEqual(entry['tags'], None)

    def test_val_data_not_null(self):
        for entry in self.app.get_val_data_dict():
            self.assertNotEqual(entry['uuid'], None)
            self.assertNotEqual(entry['postText'], None)
            self.assertNotEqual(entry['targetParagraphs'], None)
            self.assertNotEqual(entry['targetTitle'], None)
            self.assertNotEqual(entry['spoiler'], None)
            self.assertNotEqual(entry['spoilerPositions'], None)
            self.assertNotEqual(entry['tags'], None)

    def test_aug_con_train_data_not_null(self):
        for entry in self.app.get_aug_con_train_data_dict():
            self.assertNotEqual(entry['uuid'], None)
            self.assertNotEqual(entry['postText'], None)
            self.assertNotEqual(entry['targetParagraphs'], None)
            self.assertNotEqual(entry['targetTitle'], None)
            self.assertNotEqual(entry['spoiler'], None)
            self.assertNotEqual(entry['spoilerPositions'], None)
            self.assertNotEqual(entry['tags'], None)

    def test_aug_syn_train_data_not_null(self):
        for entry in self.app.get_aug_syn_train_data_dict():
            self.assertNotEqual(entry['uuid'], None)
            self.assertNotEqual(entry['postText'], None)
            self.assertNotEqual(entry['targetParagraphs'], None)
            self.assertNotEqual(entry['targetTitle'], None)
            self.assertNotEqual(entry['spoiler'], None)
            self.assertNotEqual(entry['spoilerPositions'], None)
            self.assertNotEqual(entry['tags'], None)

    def test_aug_con_val_data_not_null(self):
        for entry in self.app.get_aug_con_val_data_dict():
            self.assertNotEqual(entry['uuid'], None)
            self.assertNotEqual(entry['postText'], None)
            self.assertNotEqual(entry['targetParagraphs'], None)
            self.assertNotEqual(entry['targetTitle'], None)
            self.assertNotEqual(entry['spoiler'], None)
            self.assertNotEqual(entry['spoilerPositions'], None)
            self.assertNotEqual(entry['tags'], None)

    def test_aug_syn_val_data_not_null(self):
        for entry in self.app.get_aug_syn_val_data_dict():
            self.assertNotEqual(entry['uuid'], None)
            self.assertNotEqual(entry['postText'], None)
            self.assertNotEqual(entry['targetParagraphs'], None)
            self.assertNotEqual(entry['targetTitle'], None)
            self.assertNotEqual(entry['spoiler'], None)
            self.assertNotEqual(entry['spoilerPositions'], None)
            self.assertNotEqual(entry['tags'], None)

    def test_aug_con_train_data_target_paragraphs_length(self):
        for entry_aug, entry in zip(self.app.get_aug_con_train_data_dict(), self.app.get_train_data_dict()):
            self.assertTrue(len("\n".join(entry_aug['targetParagraphs']).split(" ")) >=
                            len("\n".join(entry['targetParagraphs']).split(" ")))

    def test_aug_syn_train_data_target_paragraphs_length(self):
        for entry_aug, entry in zip(self.app.get_aug_syn_train_data_dict(), self.app.get_train_data_dict()):
            self.assertTrue(len("\n".join(entry_aug['targetParagraphs']).split(" ")) >=
                            len("\n".join(entry['targetParagraphs']).split(" ")))

    def test_aug_con_val_data_target_paragraphs_length(self):
        for entry_aug, entry in zip(self.app.get_aug_con_val_data_dict(), self.app.get_val_data_dict()):
            self.assertTrue(len("\n".join(entry_aug['targetParagraphs']).split(" ")) >=
                            len("\n".join(entry['targetParagraphs']).split(" ")))

    def test_aug_syn_val_data_target_paragraphs_length(self):
        for entry_aug, entry in zip(self.app.get_aug_syn_val_data_dict(), self.app.get_val_data_dict()):
            self.assertTrue(len("\n".join(entry_aug['targetParagraphs']).split(" ")) >=
                            len("\n".join(entry['targetParagraphs']).split(" ")))


if __name__ == '_main_':
    unittest.main()
