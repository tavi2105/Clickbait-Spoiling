from modele.Clickbait import Clickbait
from modele.ClickbaitSummaryType import ClickbaitSummaryType


class ClickbaitSolved(Clickbait):
    def __init__(self):
        super().__init__()
        self.humanSpoiler: str
        self.spoiler: str
        self.spoiler_position: [int][int]
        self.type: ClickbaitSummaryType
