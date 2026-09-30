"""Minimal end-to-end example: FD discovery on a small table.

Run with:  PYTHONPATH=src python examples/quickstart.py
(uses the pure-Python reference miner, so no C++ build is needed)
"""
import pandas as pd

import nesydep as nd

df = pd.DataFrame(
    {
        "zip": ["10001", "10001", "90001", "90001", "60601", "60601"],
        "city": ["New York", "New York", "Los Angeles", "Los Angeles", "Chicago", "Chicago"],
        "state": ["NY", "NY", "CA", "CA", "IL", "IL"],
        "name": ["Ann", "Bob", "Cat", "Dan", "Eve", "Fox"],
    }
)

result = nd.discover(df, algo="pyref-fd")
print(f"discovered {len(result)} FDs:")
for fd in result.fds:
    if "name" not in fd.lhs and fd.rhs != "name":  # hide the key column's FDs
        print(" ", fd)
