from typing import Any, Callable, Dict, Optional
from abc import ABCMeta, abstractmethod


class Container:
    def __init__(self, elements: Optional[Dict[str, Any]] = None):
        self.elements = elements if elements is not None else dict()

    def __getitem__(self, item):
        return self.elements[item]


class Stage(metaclass=ABCMeta):
    def __init__(self, config):
        self.__config = config

    def __call__(self, arg: Optional[Container] = None) -> Container:
        return self.call(self.config, arg)

    @property
    def config(self):
        return self.__config

    @config.setter
    def config(self, value: Dict[str, Any]):
        for k, v in value.items():
            if k in self.__config:
                self.config[k] = v

    @property
    @abstractmethod
    def name(self) -> str:
        return NotImplemented

    @abstractmethod
    def call(self, config: Dict[str, Any], arg: Container) -> Container:
        return NotImplemented


def stage(configuration: Dict[str, Any]):
    def decorator(function: Callable[[Dict[str, Any], Optional[Container]], Container]):
        class Wrapper(Stage):
            @property
            def name(self) -> str:
                return function.__name__

            def call(self, config: Dict[str, Any], arg: Container) -> Container:
                return function(config, arg)

        return Wrapper(configuration)
    return decorator
