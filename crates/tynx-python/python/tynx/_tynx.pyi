from os import PathLike
from typing import final

__all__ = ["Session", "__version__"]
__version__: str

@final
class Session:
    def __new__(cls, path: str | PathLike[str], *, simplify: bool = True) -> Session: ...
    @property
    def inputs(self) -> list[str]: ...
    @property
    def outputs(self) -> list[str]: ...
