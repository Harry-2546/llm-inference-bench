#!/usr/bin/env python3
"""fix66277imports.py FILE : add the imports that vllm#66277 needs to vllm/utils/b12x.py
(the only #66277 hunk that does not apply on top of the #68474 stack)."""
import sys

p = sys.argv[1]
s = open(p).read()
a = "from collections.abc import Callable, Hashable, Iterable\n"
assert s.count(a) == 1, "collections anchor"
s = s.replace(a, "from collections.abc import Callable, Hashable, Iterable, Iterator\nfrom contextlib import contextmanager\n", 1)
a = "from vllm.logger import init_logger\n"
assert s.count(a) == 1, "logger anchor"
s = s.replace(a, a + "from vllm.platforms import current_platform\n", 1)
open(p, "w").write(s)
print("ok")
