class Clickbait:
    def __init__(self, uuid, postText, targetParagraphs, targetTitle, targetUrl, spoiler='', spoilerPositions='', tags=[],
                 provenance='', postId='', postPlatform='', targetDescription='', targetKeywords='', targetMedia=''):
        self.targetUrl: str = targetUrl
        self.targetTitle: str = targetTitle
        self.targetParagraphs: str = targetParagraphs
        self.uuid: str = uuid
        self.postText: str = postText

        