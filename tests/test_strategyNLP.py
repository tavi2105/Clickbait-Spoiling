import unittest

import pandas
from sklearn.pipeline import Pipeline

from Preliminary_classes.classification_algoritms.NaiveBayes import NaiveBayes
from Preliminary_classes.classification_algoritms.LogisticRegression import LogisticRegression
from Preliminary_classes.classification_algoritms.SVM import SVM
from tests.test_utils import TRAINING_DATA, EVALUATING_DATA, SUMMARY_TYPES, is_fitted


class TestingStrategyForClassification(unittest.TestCase):

    def setUp(self):
        # aici ne setam lista de algoritmi pe care vrem sa-i testam
        self.algorithms = [NaiveBayes(), LogisticRegression(), SVM()]

    def test_prepare_training_data(self):
        for a in self.algorithms:
            with self.subTest(type(a).__name__ + 'prepare training data'):
                rez = a.prepare_data([TRAINING_DATA[0]])
                pandas.testing.assert_frame_equal(rez, pandas.DataFrame(
                    [{"targetParagraphs": "Something happened\nIt was fantastic\nEverybody love it", "type": 2}]))

    def test_prepare_evaluating_data(self):
        for a in self.algorithms:
            with self.subTest(type(a).__name__ + ' prepare evaluating data'):
                rez = a.prepare_data([EVALUATING_DATA[0]])
                pandas.testing.assert_frame_equal(rez, pandas.DataFrame(
                    [{"targetParagraphs": "Something happened\nIt was fantastic\nEverybody love it"}]))

    def test_training_take_place(self):
        for a in self.algorithms:
            with self.subTest(type(a).__name__ + ' training take place'):
                a.train(TRAINING_DATA)
                self.assertEqual(is_fitted(a.model['clf']), True)

    def test_apply_on_single_clickbait(self):
        for a in self.algorithms:
            with self.subTest(type(a).__name__ + ' applying on single clickbait'):
                a.train(TRAINING_DATA)
                rez = a.apply_on_single_clickbait(EVALUATING_DATA[0])
                self.assertIn(rez, SUMMARY_TYPES)

    def test_apply_on_list_of_clickbaits(self):
        for a in self.algorithms:
            with self.subTest(type(a).__name__ + ' applying on list of clickbaits'):
                a.train(TRAINING_DATA)
                rez = a.apply_on_list_of_clickbaits(EVALUATING_DATA)
                areAllItemsOk = all(el in SUMMARY_TYPES for el in rez)
                self.assertEqual(areAllItemsOk and len(rez) == len(EVALUATING_DATA), True)


if __name__ == '__main__':
    unittest.main()
