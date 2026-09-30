"""Built-in miners (importing registers them)."""
from nesydep.miners.pyref import NaiveFDMiner  # noqa: F401
from nesydep.miners.native import (  # noqa: F401
    CTaneNative,
    DFDNative,
    PFMinerNative,
    SCFDMNative,
    TaneNative,
)
