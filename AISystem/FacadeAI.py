import DataExtractor
import AI


class FacadeAI:
    """
    The Facade class provides a simple interface to the complex logic of one or
    several subsystems.


    All requests from the Backend will be managed by the Facade.

    Facade & Command: Backend use the Command to send requests to the AISystem. The Command send simple requests
    to the Facade. The Facade "knows" the AISystem logic and it makes the whole process for one operation step by step.
    """

    def __init__(self) -> None:
        """
        Depending on your application's needs, you can provide the Facade with
        existing subsystem objects or force the Facade to create them on its
        own.
        """

        self._data = DataExtractor
        self._ai = AI

    def apply_synopsys(self, clickbait) -> str:

        # TODO: Step 1: process given data (link -> Clickbait format)
        #       Step 2: check if training was made:
        #                   - yes: -> apply trained model on data
        #                   - no:  -> train & evaluate data, then apply the trained model on the new data
        #       Step 3: Process the spoiler and return it to the Backend

        pass

    def feedback(self, feedback) -> str:

        # TODO: Step 1: process given data (text -> Clickbait format)
        #       Step 2: check if text is a valid clickbait (contains all fields that was required with a valid type):
        #                   - yes: -> add clickbait to the training/evaluate dateset
        #                             & the Observer will trigger training
        #                   - no:  -> do nothing
        #       Step 3: return status to the Backend

        pass

    def trigger(self) -> str:

        # TODO: trigger a new training session

        pass


