from models.ClickbaitSummaryType import ClickbaitSummaryType


class ClickbaitSynopsis:
    def __init__(self):
        self.spoilerType: ClickbaitSummaryType
        self.spoiler: str
        self.uuid: str
