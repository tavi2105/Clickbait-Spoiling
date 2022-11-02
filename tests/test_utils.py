from Preliminary_classes.models.Clickbait import Clickbait
from Preliminary_classes.models.ClickbaitSolved import ClickbaitSolved
from Preliminary_classes.models.ClickbaitSummaryType import ClickbaitSummaryType

TRAINING_DATA = [ClickbaitSolved("1", "PostText1", ["Something happened", "It was fantastic", "Everybody love it"],
                                 "Something fantastic", "https://gamerant.com/", "Something fantastic happened",
                                 ["Something happened", "It was fantastic"], [[[0, 0], [0, 14]], [[1, 0], [1, 102]]],
                                 ClickbaitSummaryType.MULTI),
                 ClickbaitSolved("2", "PostText2", ["It was horrible", "Alan died", "He was so young"], "Bad news",
                                 "https://deadPeople.com/", "Alan died", ["Alan died"], [[[1, 0], [1, 7]]],
                                 ClickbaitSummaryType.PASSAGE),
                 ClickbaitSolved("3", "PostText3",
                                 ["The new mac appeared", "Everybody wanted to buy it", "But it is limited edition",
                                  "The price is big", "It costs 200000$"], "How much costs the new mac", "", "200000$",
                                 ["200000$"], [[[3, 9], [3, 16]]], ClickbaitSummaryType.PHRASE),
                 ClickbaitSolved("4", "PostText4", ["Something happened", "It was fantastic", "Everybody love it"],
                                 "Something fantastic", "https://gamerant.com/", "Something fantastic happened",
                                 ["Something happened", "It was fantastic"], [[[0, 0], [0, 14]], [[1, 0], [1, 102]]],
                                 ClickbaitSummaryType.MULTI),
                 ClickbaitSolved("5", "PostText5", ["It was horrible", "Alan died", "He was so young"], "Bad news",
                                 "https://deadPeople.com/", "Alan died", ["Alan died"], [[[1, 0], [1, 7]]],
                                 ClickbaitSummaryType.PASSAGE),
                 ClickbaitSolved("6", "PostText6",
                                 ["The new mac appeared", "Everybody wanted to buy it", "But it is limited edition",
                                  "The price is big", "It costs 200000$"], "How much costs the new mac", "", "200000$",
                                 ["200000$"], [[[3, 9], [3, 16]]], ClickbaitSummaryType.PHRASE),
                 ClickbaitSolved("7", "PostText7", ["Something happened", "It was fantastic", "Everybody love it"],
                                 "Something fantastic", "https://gamerant.com/", "Something fantastic happened",
                                 ["Something happened", "It was fantastic"], [[[0, 0], [0, 14]], [[1, 0], [1, 102]]],
                                 ClickbaitSummaryType.MULTI),
                 ClickbaitSolved("8", "PostText8", ["It was horrible", "Alan died", "He was so young"], "Bad news",
                                 "https://deadPeople.com/", "Alan died", ["Alan died"], [[[1, 0], [1, 7]]],
                                 ClickbaitSummaryType.PASSAGE),
                 ClickbaitSolved("9", "PostText9",
                                 ["The new mac appeared", "Everybody wanted to buy it", "But it is limited edition",
                                  "The price is big", "It costs 200000$"], "How much costs the new mac", "", "200000$",
                                 ["200000$"], [[[3, 9], [3, 16]]], ClickbaitSummaryType.PHRASE),
                 ]

VALIDATING_DATA = [ClickbaitSolved("10", "PostText10", ["Something happened", "It was fantastic", "Everybody love it"],
                                   "Something fantastic", "https://gamerant.com/", "Something fantastic happened",
                                   ["Something happened", "It was fantastic"], [[[0, 0], [0, 14]], [[1, 0], [1, 102]]],
                                   ClickbaitSummaryType.MULTI),
                   ClickbaitSolved("11", "PostText11", ["It was horrible", "Alan died", "He was so young"], "Bad news",
                                   "https://deadPeople.com/", "Alan died", ["Alan died"], [[[1, 0], [1, 7]]],
                                   ClickbaitSummaryType.PASSAGE),
                   ClickbaitSolved("12", "PostText12",
                                   ["The new mac appeared", "Everybody wanted to buy it", "But it is limited edition",
                                    "The price is big", "It costs 200000$"], "How much costs the new mac", "",
                                   "200000$", ["200000$"], [[[3, 9], [3, 16]]], ClickbaitSummaryType.PHRASE),
                   ]
EVALUATING_DATA = [Clickbait("13", "PostText13", ["Something happened", "It was fantastic", "Everybody love it"],
                             "Something fantastic", "https://gamerant.com/"),
                   Clickbait("14", "PostText14",
                             ["The new mac appeared", "Everybody wanted to buy it", "But it is limited edition",
                              "The price is big", "It costs 200000$"], "How much costs the new mac",
                             "https://apple.com/",
                             ),
                   ]

SUMMARY_TYPES = [t.name for t in ClickbaitSummaryType]


def is_fitted(model):
    """
    Checks if a scikit-learn estimator/transformer has already been fit.


    Parameters
    ----------
    model: scikit-learn estimator (e.g. RandomForestClassifier)
        or transformer (e.g. MinMaxScaler) object


    Returns
    -------
    Boolean that indicates if ``model`` has already been fit (True) or not (False).
    """

    attrs = [v for v in vars(model)
             if v.endswith("_") and not v.startswith("__")]

    return len(attrs) != 0
