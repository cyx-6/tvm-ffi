#!/usr/bin/env python3
"""Debug script: check if MSVC C++ runtime symbols are exported from DLLs.

This script helps diagnose Windows JIT symbol resolution failures by:
1. Enumerating all loaded modules in the current process
2. Testing GetProcAddress for specific MSVC C++ runtime symbols
3. Dumping undefined symbols from COFF object files using llvm-objdump

Run on Windows CI to check if vcruntime140.dll exports the symbols we need.
"""
import ctypes
import ctypes.wintypes
import os
import subprocess
import sys


def main():
    print("=" * 70)
    print("DEBUG: Windows Symbol Resolution Diagnostics")
    print("=" * 70)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    # --- Step 1: Enumerate all loaded modules ---
    print("\n--- Loaded Modules in Current Process ---")
    hProcess = kernel32.GetCurrentProcess()
    hMods = (ctypes.wintypes.HMODULE * 1024)()
    cbNeeded = ctypes.wintypes.DWORD()
    if psapi.EnumProcessModules(
        hProcess, ctypes.byref(hMods), ctypes.sizeof(hMods), ctypes.byref(cbNeeded)
    ):
        count = cbNeeded.value // ctypes.sizeof(ctypes.wintypes.HMODULE)
        modname = ctypes.create_unicode_buffer(260)
        for i in range(min(count, 50)):  # Print first 50 modules
            if kernel32.GetModuleFileNameW(hMods[i], modname, 260):
                print(f"  [{i}] {modname.value}")
        if count > 50:
            print(f"  ... and {count - 50} more modules")
    else:
        print(f"  EnumProcessModules failed: {ctypes.get_last_error()}")

    # --- Step 2: Test specific DLLs ---
    runtime_dlls = [
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "ucrtbase.dll",
        "msvcp140.dll",
        "msvcrt.dll",
    ]
    print("\n--- Runtime DLL Handles ---")
    dll_handles = {}
    for dll in runtime_dlls:
        h = kernel32.LoadLibraryA(dll.encode())
        dll_handles[dll] = h
        print(f"  {dll}: handle={hex(h) if h else 'FAILED'}")

    # --- Step 3: Test GetProcAddress for missing symbols ---
    target_symbols = [
        b"_Init_thread_header",
        b"_Init_thread_footer",
        b"_Init_thread_abort",
        b"??_7type_info@@6B@",
        b"??3@YAXPEAX_K@Z",
        # Also test some symbols that should definitely exist
        b"memcpy",
        b"malloc",
        b"free",
    ]

    print("\n--- GetProcAddress Results Per DLL ---")
    for dll_name, h in dll_handles.items():
        if not h:
            continue
        print(f"\n  {dll_name}:")
        for sym in target_symbols:
            addr = kernel32.GetProcAddress(h, sym)
            status = hex(addr) if addr else "NOT FOUND"
            print(f"    {sym.decode():40s} -> {status}")

    # --- Step 4: Search all loaded modules for each symbol ---
    print("\n--- Full Process Module Search ---")
    for sym in target_symbols:
        found = False
        for i in range(min(cbNeeded.value // ctypes.sizeof(ctypes.wintypes.HMODULE), 1024)):
            addr = kernel32.GetProcAddress(hMods[i], sym)
            if addr:
                modname_buf = ctypes.create_unicode_buffer(260)
                kernel32.GetModuleFileNameW(hMods[i], modname_buf, 260)
                mod_short = os.path.basename(modname_buf.value)
                print(f"  {sym.decode():40s} -> {hex(addr)} in {mod_short}")
                found = True
                break
        if not found:
            print(f"  {sym.decode():40s} -> NOT FOUND IN ANY MODULE")

    # --- Step 5: Try __imp_ prefixed versions ---
    print("\n--- GetProcAddress for __imp_ prefixed symbols ---")
    imp_symbols = [
        b"__imp__Init_thread_header",
        b"__imp__Init_thread_footer",
        b"__imp__Init_thread_abort",
        b"__imp_??_7type_info@@6B@",
        b"__imp_??3@YAXPEAX_K@Z",
    ]
    for sym in imp_symbols:
        found = False
        for dll_name, h in dll_handles.items():
            if not h:
                continue
            addr = kernel32.GetProcAddress(h, sym)
            if addr:
                print(f"  {sym.decode():50s} -> {hex(addr)} in {dll_name}")
                found = True
                break
        if not found:
            print(f"  {sym.decode():50s} -> NOT FOUND")

    # --- Step 6: Dump undefined symbols from test object files ---
    print("\n--- Undefined Symbols in Test Object Files ---")
    # Try to find llvm-objdump
    llvm_objdump = None
    for candidate in [
        r"C:\opt\llvm\Library\bin\llvm-objdump.exe",
        r"C:\opt\llvm\bin\llvm-objdump.exe",
    ]:
        if os.path.exists(candidate):
            llvm_objdump = candidate
            break

    if not llvm_objdump:
        # Try to find it in PATH
        try:
            result = subprocess.run(
                ["where", "llvm-objdump"], capture_output=True, text=True
            )
            if result.returncode == 0:
                llvm_objdump = result.stdout.strip().split("\n")[0]
        except Exception:
            pass

    if llvm_objdump:
        print(f"  Using: {llvm_objdump}")
        # Find test object files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tests_dir = os.path.join(script_dir, "..", "tests")
        for obj_name in ["test_funcs.o", "test_ctor_dtor.o"]:
            obj_path = os.path.join(tests_dir, obj_name)
            if os.path.exists(obj_path):
                print(f"\n  {obj_name} - undefined symbols:")
                try:
                    result = subprocess.run(
                        [llvm_objdump, "-t", obj_path],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    for line in result.stdout.splitlines():
                        if "*UND*" in line or "UNDEF" in line:
                            print(f"    {line.strip()}")
                except Exception as e:
                    print(f"    Error: {e}")
            else:
                print(f"\n  {obj_name}: NOT FOUND at {obj_path}")
    else:
        print("  llvm-objdump not found, skipping object file analysis")

    # --- Step 7: Try dumpbin for DLL exports ---
    print("\n--- vcruntime140.dll Export Check (via dumpbin) ---")
    try:
        result = subprocess.run(
            ["dumpbin", "/exports", r"C:\Windows\System32\vcruntime140.dll"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Filter for our target symbols
            for line in result.stdout.splitlines():
                for sym in [
                    "Init_thread",
                    "type_info",
                    "??3@",
                    "operator delete",
                ]:
                    if sym in line:
                        print(f"  {line.strip()}")
                        break
        else:
            print(f"  dumpbin failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print("  dumpbin not found in PATH")
    except Exception as e:
        print(f"  Error: {e}")

    # --- Step 8: Check Python's own runtime DLL linkage ---
    print("\n--- Python Runtime Info ---")
    print(f"  Python: {sys.version}")
    print(f"  Executable: {sys.executable}")
    print(f"  Platform: {sys.platform}")

    print("\n" + "=" * 70)
    print("DEBUG: Diagnostics complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
