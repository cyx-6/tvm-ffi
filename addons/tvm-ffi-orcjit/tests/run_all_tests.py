#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Build test objects, run pytest, and run quick-start examples.

Single entry point for CI.  Replaces the separate cmake build step and
platform-specific test commands.

Usage:
    python run_all_tests.py [--llvm-prefix /opt/llvm]
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
QUICKSTART_DIR = TESTS_DIR.parent / "examples" / "quick-start"


def _run(cmd: list[str], **kwargs: object) -> None:
    print(f"+ {' '.join(str(a) for a in cmd)}", flush=True)
    subprocess.check_call(cmd, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llvm-prefix", default=None,
                        help="LLVM install prefix (forwarded to build_test_objects)")
    args = parser.parse_args()

    # 1. Build test objects + quick-start example
    from build_test_objects import ensure_built
    ensure_built(args.llvm_prefix)

    # 2. Run pytest
    print(f"\n{'='*60}\nRunning pytest\n{'='*60}\n", flush=True)
    _run([sys.executable, "-m", "pytest", str(TESTS_DIR), "-v"])

    # 3. Run quick-start examples
    print(f"\n{'='*60}\nRunning quick-start examples\n{'='*60}\n", flush=True)
    langs = ["c"]
    if platform.system() != "Windows":
        langs.insert(0, "cpp")
    for lang in langs:
        _run(
            [sys.executable, str(QUICKSTART_DIR / "run.py"), "--lang", lang],
            cwd=str(QUICKSTART_DIR),
        )

    print(f"\n{'='*60}\nAll tests passed\n{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
