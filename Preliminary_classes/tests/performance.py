import cProfile
import datetime
from AI import AI
from models.Clickbait import Clickbait
from main import main
from data_augmentation.Synonym import main_aug


def test_ai():
    cProfile.run("ai = AI()")


def test_synopsys():
    cProfile.run("main()")


def syn_aug():
    cProfile.run("main_aug()")


if __name__ == "__main__":
    syn_aug()
