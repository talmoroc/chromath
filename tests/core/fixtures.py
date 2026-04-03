import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace

_FIXTURES_PATH = Path(__file__).parent / "fixtures.json"


def _load() -> SimpleNamespace:
    with open(_FIXTURES_PATH) as f:
        raw = json.load(f)

    ns = SimpleNamespace()
    for category, items in raw.items():
        cat_ns = SimpleNamespace()
        for name, vector in items.items():
            setattr(cat_ns, name, np.array(vector, dtype=np.bool_))
        setattr(ns, category, cat_ns)
    return ns


fixtures = _load()
