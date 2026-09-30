"""Hand-verified ground truth for ``toy_addresses.csv``.

The toy table satisfies these minimal non-trivial FDs exactly:

- zip -> city            (each ZIP maps to one city)
- zip -> state           (each ZIP maps to one state)
- city -> state          (each city belongs to one state)
- id -> <every column>   (id is a key)

And these exact CFDs hold:

- [city] => state, (New York || NY)   etc. — constant CFDs per city
- [zip] => city,  (_ || _)            variable CFD (pattern wildcard)
"""

from __future__ import annotations

KEY_COLUMN = "id"

MINIMAL_FDS = {
    (("zip",), "city"),
    (("zip",), "state"),
    (("city",), "state"),
}

# id is a key: it functionally determines every other column.
KEY_FDS = {(("id",), c) for c in ("zip", "city", "state", "gender", "age_group")}

ALL_MINIMAL_FDS = MINIMAL_FDS | KEY_FDS
