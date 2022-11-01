from typing import List

from interfaces import Observer
from interfaces.SubjectToObserve import SubjectToObserve


class UpdateNotifier(SubjectToObserve):
    """
        This class implement the SubjectToObserve interface.
        It contains the number of correct received feedbacks and everytime the number increases with
        a certain amount it will notify the observers attached
        """

    def __init__(self, max_amount):
        self._state: int = 0
        """
        The Subject's state, essential to all subscribers, is stored in this variable, representing
        the number of correct received feedbacks. In the final code, it will be initialized with a value read
        from database
        """

        self._amount_needed_for_notify = max_amount
        """
        The threshold the state needs to reach so that the subject to notify its subscribers
        """

        self._observers: List[Observer] = []
        """
        List of subscribers.
        """

    def attach(self, observer: Observer) -> None:
        print("Subject: Attached an observer.")
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    """
    The subscription management methods.
    """

    def notify(self) -> None:
        """
        Trigger an update in each subscriber if the condition is fulfilled
        """
        if self._state > self._amount_needed_for_notify:
            print("Subject: Notifying observers...")
            for observer in self._observers:
                observer.update(self)
        
