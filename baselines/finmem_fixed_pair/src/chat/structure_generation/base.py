from abc import ABC, abstractmethod
from typing import List, Union

from ...utils import RunMode


class BaseStructureGenerationSchema(ABC):
    @staticmethod
    @abstractmethod
    def __call__(
        run_mode: RunMode,
        short_memory_ids: Union[List[int], None] = None,
        mid_memory_ids: Union[List[int], None] = None,
        long_memory_ids: Union[List[int], None] = None,
        reflection_memory_ids: Union[List[int], None] = None,
    ):
        pass
