from abc import ABC, abstractmethod

class MainEntrypoint(ABC):

    @abstractmethod
    def run(self):
        pass