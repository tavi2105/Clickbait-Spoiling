import cProfile
import pstats
from pstats import SortKey
from AI import AI
from models.Clickbait import Clickbait

prof = cProfile.Profile()


def main():
    sample_clickbait = Clickbait("13", "NASA sets date for full recovery of ozone hole",
                                 ["2070 is shaping up to be a great year for Mother Earth.",
                                  "That's when NASA scientists are predicting the hole in the ozone layer might "
                                  "finally make a full recovery. Researchers announced their conclusion, in addition "
                                  "to other findings, in a presentation Wednesday during the annual American "
                                  "Geophysical Union meeting in San Francisco.",
                                  "The team of scientists specifically looked at the chemical composition of the "
                                  "ozone hole, which has shifted in both size and depth since the passing of the "
                                  "Montreal Protocol in 1987. The agreement banned its 197 signatory countries from "
                                  "using chemicals, like chlorofluorocarbons (CFCs), that break down into chlorine in "
                                  "the upper atmosphere and harm the ozone layer.",
                                  "They found that, while levels of chlorine in the atmosphere have indeed decreased "
                                  "as a result of the protocol, it's too soon to tie them to a healthier ozone layer.",
                                  "\"Ozone holes with smaller areas and a larger total amount of ozone are not "
                                  "necessarily evidence of recovery attributable to the expected chlorine decline,"
                                  "\" Susan Strahan of NASA's Goddard Space Flight Center explained in a media "
                                  "briefing. \"That assumption is like trying to understand what's wrong with your "
                                  "car's engine without lifting the hood.\"",
                                  "Instead, the scientists believe the most recent ozone hole changes, including both "
                                  "the largest hole ever, in 2006, and one of the smallest holes, in 2012, "
                                  "are primarily due to weather. Strong winds have the ability to move ozone in large "
                                  "quantities, effectively blocking the hole some years, while failing to block it in "
                                  "others.",
                                  "\"At the moment, it is winds and temperatures that are really controlling how big "
                                  "[the ozone hole] is,\" Strahan told the BBC.",
                                  "LiveScience reports weather is expected to be the predominant factor in the ozone "
                                  "hole's size until 2025, at which point CFCs will have dropped enough as a result "
                                  "of the Montreal Protocol to become noticeable.",
                                  "By 2070, however, the ozone hole is expected to have made a full recovery.",
                                  "\"It’s not going to be a smooth ride,\" Strahan cautioned the Los Angeles Times. "
                                  "\"There will be some bumps in the road, but overall the trend is downward.\""],
                                 "Hole In Ozone Layer Expected To Make Full Recovery By 2070: NASA", "https"
                                                                                                     "://gamerant.com/")
    prof.enable()
    ai = AI()
    print(ai.generate_clickbait_synopsis(sample_clickbait))
    prof.disable()


if __name__ == "__main__":
    main()
    # retrieve the stats
    ps = pstats.Stats(prof)
    # Remove directory path form file names
    ps.strip_dirs()
    # Sort by the cumulative time to know what function need performance improvements
    ps.sort_stats(SortKey.CUMULATIVE)
    # Show performance statistics just for the functions we implemented
    ps.print_stats(
        'qa|LogisticRegression|NaiveBayes|SVM|eval_for_classification|eval_for_summarization|SynopsisGenerator'
        '|DataExtractor|ModelSelector|AI')
