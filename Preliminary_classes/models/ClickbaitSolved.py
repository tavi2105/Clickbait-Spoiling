from models.Clickbait import Clickbait
from models.ClickbaitSummaryType import ClickbaitSummaryType


class ClickbaitSolved(Clickbait):
    def __init__(self, uuid, postText, targetParagraphs, targetTitle, targetUrl, spoiler, spoilerPositions, tags,
                 provenance='', postId='', postPlatform='', targetDescription='', targetKeywords='', targetMedia=''):
        super().__init__(uuid, postText, targetParagraphs, targetTitle, targetUrl)
        self.summary_type: ClickbaitSummaryType = ClickbaitSummaryType.MULTI \
            if tags[0] == 'multi' else ClickbaitSummaryType.PHRASE \
            if tags[0] == 'phrase' else ClickbaitSummaryType.PASSAGE
        self.spoiler_position: [int][int] = spoilerPositions
        self.spoiler: str = spoiler
