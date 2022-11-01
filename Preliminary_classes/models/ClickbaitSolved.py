from Preliminary_classes.models.Clickbait import Clickbait
from Preliminary_classes.models.ClickbaitSummaryType import ClickbaitSummaryType


class ClickbaitSolved(Clickbait):
    def __init__(self, uuid, postText, targetParagraphs, targetTitle, targetUrl,
                 humanSpoiler, spoiler, spoiler_position, summary_type):
        super().__init__(uuid, postText, targetParagraphs, targetTitle, targetUrl)
        self.summary_type: ClickbaitSummaryType = summary_type
        self.spoiler_position: [int][int] = spoiler_position
        self.spoiler: str = spoiler
        self.humanSpoiler: str = humanSpoiler

