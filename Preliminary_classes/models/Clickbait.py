class Clickbait:
    def __init__(self, uuid, postText, targetParagraphs, targetTitle, targetUrl):
        self.targetUrl: str = targetUrl
        self.targetTitle: str = targetTitle
        self.targetParagraphs: str = targetParagraphs
        self.uuid: str = uuid
        self.postText: str = postText

        