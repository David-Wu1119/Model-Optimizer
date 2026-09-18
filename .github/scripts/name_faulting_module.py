"""Name the module that executed the illegal instruction, from a minidump.

The Python-level faulthandler cannot do this: it prints interpreter frames, and the fault is in
native code several frames below the interpreter. The minidump carries the exception record and
the loaded-module list, so the faulting address can be resolved to a DLL by containment.
"""

import sys
from pathlib import Path


def main(folder: str) -> int:
    dumps = sorted(Path(folder).glob("*.dmp"))
    if not dumps:
        print("no .dmp files -- the crash did not reach the postmortem debugger")
        return 0
    try:
        from minidump.minidumpfile import MinidumpFile
    except ImportError:
        print("minidump package unavailable; dumps are uploaded as an artifact instead")
        return 0

    for d in dumps:
        print(f"=== {d.name} ({d.stat().st_size / 1e6:.1f} MB) ===")
        try:
            mf = MinidumpFile.parse(str(d))
        except Exception as exc:  # noqa: BLE001 - diagnostic, never fatal
            print(f"  unreadable: {exc}")
            continue

        addr = None
        exc_rec = getattr(mf, "exception", None)
        if exc_rec is not None:
            for record in getattr(exc_rec, "exception_records", []) or [exc_rec]:
                er = getattr(record, "ExceptionRecord", record)
                code = getattr(er, "ExceptionCode", None)
                addr = getattr(er, "ExceptionAddress", None)
                code_s = f"0x{int(code):08x}" if isinstance(code, int) else str(code)
                print(f"  exception {code_s} at 0x{addr:x}" if addr else f"  exception {code_s}")
                break

        mods = getattr(getattr(mf, "modules", None), "modules", []) or []
        print(f"  {len(mods)} modules loaded")
        if addr is None:
            continue
        for m in mods:
            base = getattr(m, "baseaddress", 0)
            size = getattr(m, "size", 0)
            if base <= addr < base + size:
                print(f"  >>> FAULTING MODULE: {getattr(m, 'name', '?')}  (+0x{addr - base:x})")
                break
        else:
            print(f"  >>> address 0x{addr:x} is in NO loaded module "
                  "-- a jump into non-code memory, i.e. corruption, not a missing opcode")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "crashdumps"))
