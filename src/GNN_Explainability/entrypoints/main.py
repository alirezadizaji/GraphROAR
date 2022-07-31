from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.base_config import BaseConfig

class MainEntrypoint(ABC):
    def __init__(self, conf: 'BaseConfig') -> None:
        self.conf = conf
        self.conf.set_loaders()

    @abstractmethod
    def run(self):
        pass