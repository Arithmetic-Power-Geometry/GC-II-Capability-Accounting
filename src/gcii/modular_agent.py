from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import random

CHANNELS = ("R", "I", "A", "L")

@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    payload: tuple
    answer: object
    required: frozenset[str]


def _subset_label(s):
    return "base" if not s else "".join(sorted(s))


def generate_tasks(seed: int = 20260908, per_requirement: int = 25):
    rng = random.Random(seed)
    reqs = [frozenset()]
    for r in range(1, 5):
        reqs += [frozenset(x) for x in combinations(CHANNELS, r)]
    tasks = []
    for req in reqs:
        for j in range(per_requirement):
            label = _subset_label(req)
            a = rng.randint(3, 50)
            b = rng.randint(2, 25)
            c = rng.randint(1, 20)
            if not req:
                ans = a + b
                payload = (a, b, 0, "base")
            else:
                x = a + b
                if "R" in req:
                    x += sum((a + t) % 7 for t in range(1, 5))
                if "I" in req:
                    x += c
                if "A" in req:
                    x = x * ((b % 4) + 2)
                if "L" in req:
                    x = (x * 3 + a - b) % 997
                ans = x
                payload = (a, b, c, tuple(sorted(req)))
            tasks.append(Task(f"{label}-{j:03d}", "modular", payload, ans, req))
    return tasks


def solve(task: Task, enabled: frozenset[str]):
    a, b, c, req_tuple = task.payload
    req = frozenset(req_tuple) if isinstance(req_tuple, tuple) else frozenset()
    if not req:
        return a + b, 1
    if not req.issubset(enabled):
        return None, 1
    x = a + b
    cost = 1
    if "R" in req:
        x += sum((a + t) % 7 for t in range(1, 5)); cost += 4
    if "I" in req:
        x += c; cost += 1
    if "A" in req:
        x = x * ((b % 4) + 2); cost += 1
    if "L" in req:
        x = (x * 3 + a - b) % 997; cost += 1
    return x, cost
