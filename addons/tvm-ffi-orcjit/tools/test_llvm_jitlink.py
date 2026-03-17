#!/usr/bin/env python3
"""Run LLVM's official ORC JIT test cases via llvm-jitlink on Windows CI.

This verifies that the conda-installed LLVM + ORC runtime (COFFPlatform)
works correctly on Windows before running our own tvm-ffi-orcjit tests.

Usage:
    python test_llvm_jitlink.py [--llvm-prefix C:/opt/llvm/Library]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap

# Test cases adapted from:
# llvm-project/compiler-rt/test/orc/TestCases/Windows/x86-64/

TESTS = {
    "hello-world.c": {
        "source": textwrap.dedent("""\
            #include <stdio.h>
            int main(int argc, char *argv[]) {
              printf("Hello, world!\\n");
              fflush(stdout);
              return 0;
            }
        """),
        "lang": "c",
        "extra_flags": [],
        "check_stdout": "Hello, world!",
        "jitlink_args": [],
    },
    "trivial-atexit.c": {
        "source": textwrap.dedent("""\
            #include <stdio.h>
            #include <stdlib.h>
            void meow() {
              printf("Meow\\n");
              fflush(stdout);
            }
            int main(int argc, char *argv[]) {
              atexit(meow);
              printf("Entering main\\n");
              fflush(stdout);
              return 0;
            }
        """),
        "lang": "c",
        "extra_flags": [],
        "check_stdout": "Entering main",
        "jitlink_args": [],
    },
    "static-initializer.c": {
        "source": textwrap.dedent("""\
            #include <stdio.h>
            int init_val = 0;
            #pragma section(".CRT$XIV", long, read)
            void init1(void) { init_val += 10; }
            __declspec(allocate(".CRT$XIV")) void (*p_init1)(void) = init1;
            #pragma section(".CRT$XIW", long, read)
            void init2(void) { init_val += 20; }
            __declspec(allocate(".CRT$XIW")) void (*p_init2)(void) = init2;
            int main(int argc, char *argv[]) {
              printf("init_val = %d\\n", init_val);
              fflush(stdout);
              return (init_val == 30) ? 0 : 1;
            }
        """),
        "lang": "c",
        "extra_flags": [],
        "check_stdout": "init_val = 30",
        "jitlink_args": [],
    },
}


def find_tool(llvm_prefix, name):
    """Find an LLVM tool in the prefix."""
    for subdir in ["bin", ""]:
        path = os.path.join(llvm_prefix, subdir, name)
        if os.path.isfile(path):
            return path
        path_exe = path + ".exe"
        if os.path.isfile(path_exe):
            return path_exe
    return None


def find_orc_rt(llvm_prefix):
    """Find the ORC runtime library."""
    import glob

    patterns = [
        os.path.join(llvm_prefix, "lib", "clang", "*", "lib", "windows", "orc_rt-x86_64.lib"),
        os.path.join(llvm_prefix, "lib", "clang", "*", "lib", "orc_rt-x86_64.lib"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def run_test(name, test, clang_cl, llvm_jitlink, orc_rt, tmpdir):
    """Compile and run a single test case. Returns (passed, detail)."""
    ext = ".c" if test["lang"] == "c" else ".cpp"
    src_path = os.path.join(tmpdir, name.replace("/", "_"))
    obj_path = src_path + ".o"

    with open(src_path, "w") as f:
        f.write(test["source"])

    # Step 1: Compile
    compile_cmd = [clang_cl, "-MD", "-c", "-o", obj_path] + test["extra_flags"] + [src_path]
    print(f"  Compile: {' '.join(compile_cmd)}")
    result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return False, f"Compile failed:\n{result.stderr}"

    # Step 2: Run via llvm-jitlink
    jitlink_cmd = [llvm_jitlink]
    if orc_rt:
        jitlink_cmd += ["-orc-runtime", orc_rt]
    jitlink_cmd += test["jitlink_args"] + [obj_path]
    print(f"  JITLink: {' '.join(jitlink_cmd)}")
    result = subprocess.run(jitlink_cmd, capture_output=True, text=True, timeout=30)

    output = result.stdout + result.stderr
    if result.returncode != 0:
        return False, f"llvm-jitlink failed (exit {result.returncode}):\n{output}"

    # Step 3: Check output
    expected = test.get("check_stdout")
    if expected and expected not in output:
        return False, f"Expected '{expected}' in output, got:\n{output}"

    return True, output.strip()


def main():
    parser = argparse.ArgumentParser(description="Run LLVM ORC JIT tests on Windows")
    parser.add_argument(
        "--llvm-prefix",
        default=r"C:\opt\llvm\Library",
        help="Path to LLVM installation prefix",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LLVM ORC JIT (llvm-jitlink) Verification Tests")
    print("=" * 60)

    llvm_prefix = args.llvm_prefix
    print(f"LLVM prefix: {llvm_prefix}")

    # Find tools
    clang_cl = find_tool(llvm_prefix, "clang-cl")
    llvm_jitlink = find_tool(llvm_prefix, "llvm-jitlink")
    orc_rt = find_orc_rt(llvm_prefix)

    print(f"clang-cl:     {clang_cl or 'NOT FOUND'}")
    print(f"llvm-jitlink: {llvm_jitlink or 'NOT FOUND'}")
    print(f"orc_rt:       {orc_rt or 'NOT FOUND'}")

    if not clang_cl:
        print("ERROR: clang-cl not found")
        return 1
    if not llvm_jitlink:
        print("ERROR: llvm-jitlink not found")
        return 1

    print()
    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, test in TESTS.items():
            print(f"--- {name} ---")
            try:
                ok, detail = run_test(name, test, clang_cl, llvm_jitlink, orc_rt, tmpdir)
            except subprocess.TimeoutExpired:
                ok, detail = False, "TIMEOUT"
            except Exception as e:
                ok, detail = False, str(e)

            status = "PASS" if ok else "FAIL"
            print(f"  Result: {status}")
            if detail:
                for line in detail.split("\n")[:10]:
                    print(f"    {line}")
            print()

            if ok:
                passed += 1
            else:
                failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    # Return 0 even if tests fail — this is diagnostic, not gating
    return 0


if __name__ == "__main__":
    sys.exit(main())
