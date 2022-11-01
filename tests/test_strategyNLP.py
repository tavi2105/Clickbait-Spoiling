import unittest

import pandas
from sklearn.pipeline import Pipeline

from Preliminary_classes.classification_algoritms.NaiveBayes import NaiveBayes
from Preliminary_classes.classification_algoritms.LogisticRegression import LogisticRegression
from Preliminary_classes.classification_algoritms.SVM import SVM
from Preliminary_classes.models.ClickbaitSummaryType import ClickbaitSummaryType
from tests.dataForTests import TRAINING_DATA, VALIDATING_DATA, EVALUATING_DATA

summaryTypes = [t.name for t in ClickbaitSummaryType]


class TestingStrategyForClassification(unittest.TestCase):

    def setUp(self):
        # aici ne setam lista de algoritmi pe care vrem sa-i testam
        self.algoritms = [NaiveBayes(), LogisticRegression(), SVM()]

    def test_prepare_training_data(self):
        for a in self.algoritms:
            with self.subTest(type(a).__name__ + 'prepare training data'):
                rez = a.prepare_data([TRAINING_DATA[0]])
                pandas.testing.assert_frame_equal(rez, pandas.DataFrame(
                    [{"targetParagraphs": "Something happened\nIt was fantastic\nEverybody love it", "type": 2}]))

    def test_prepare_evaluating_data(self):
        for a in self.algoritms:
            with self.subTest(type(a).__name__ + ' prepare evaluating data'):
                rez = a.prepare_data([EVALUATING_DATA[0]])
                pandas.testing.assert_frame_equal(rez, pandas.DataFrame(
                    [{"targetParagraphs": "Something happened\nIt was fantastic\nEverybody love it"}]))

    def test_training_take_place(self):
        for a in self.algoritms:
            with self.subTest(type(a).__name__ + ' training take place'):
                a.train(TRAINING_DATA)
                self.assertIsInstance(a.model, Pipeline)

    def test_apply_on_single_clickbait(self):
        for a in self.algoritms:
            with self.subTest(type(a).__name__ + ' applying on single clickbait'):
                a.train(TRAINING_DATA)
                rez = a.apply_on_single_clickbait(EVALUATING_DATA[0])
                self.assertIn(rez, summaryTypes)

    def test_apply_on_list_of_clickbaits(self):
        for a in self.algoritms:
            with self.subTest(type(a).__name__ + ' applying on list of clickbaits'):
                a.train(TRAINING_DATA)
                rez = a.apply_on_list_of_clickbaits(VALIDATING_DATA)
                areAllItemsOk = all(el in summaryTypes for el in rez)
                self.assertEqual(areAllItemsOk, True)


if __name__ == '__main__':
    unittest.main()
