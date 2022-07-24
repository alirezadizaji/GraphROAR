from abc import ABC, abstractmethod

class Entrypoint(ABC):

    @abstractmethod
    def run():
        pass