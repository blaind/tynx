from os import PathLike

__version__: str

class Session:
    def __init__(self, path: str | PathLike[str], *, simplify: bool = True) -> None: ...
    @property
    def inputs(self) -> list[str]: ...
    @property
    def outputs(self) -> list[str]: ...
