from __future__ import annotations

import ctypes
import os
import platform
import struct
import sys
from pathlib import Path


def _read_cstring(data: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(data):
        raise ValueError(f"cstring offset out of range: {offset}")
    end = data.find(b"\x00", offset)
    if end == -1:
        raise ValueError(f"unterminated cstring at offset {offset}")
    return data[offset:end].decode("ascii", errors="replace")


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _parse_sections(data: bytes, section_table_off: int, count: int) -> list[dict[str, int | str]]:
    sections: list[dict[str, int | str]] = []
    for i in range(count):
        off = section_table_off + i * 40
        name = data[off : off + 8].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        virtual_size = _u32(data, off + 8)
        virtual_address = _u32(data, off + 12)
        size_of_raw_data = _u32(data, off + 16)
        ptr_to_raw_data = _u32(data, off + 20)
        sections.append(
            {
                "name": name,
                "va": virtual_address,
                "vs": virtual_size,
                "raw_size": size_of_raw_data,
                "raw_ptr": ptr_to_raw_data,
            }
        )
    return sections


def _rva_to_offset(rva: int, sections: list[dict[str, int | str]]) -> int:
    for s in sections:
        va = int(s["va"])
        size = max(int(s["vs"]), int(s["raw_size"]))
        if va <= rva < va + size:
            return int(s["raw_ptr"]) + (rva - va)
    raise ValueError(f"RVA 0x{rva:X} not found in any section")


def get_imported_dll_names(pe_path: Path) -> list[str]:
    data = pe_path.read_bytes()

    if data[:2] != b"MZ":
        raise ValueError("Not an MZ executable")

    pe_off = _u32(data, 0x3C)
    if data[pe_off : pe_off + 4] != b"PE\x00\x00":
        raise ValueError("Invalid PE signature")

    file_header_off = pe_off + 4
    (
        _machine,
        number_of_sections,
        _time_date_stamp,
        _ptr_to_symbol_table,
        _number_of_symbols,
        size_of_optional_header,
        _characteristics,
    ) = struct.unpack_from("<HHIIIHH", data, file_header_off)

    optional_header_off = file_header_off + 20
    magic = _u16(data, optional_header_off)

    # IMAGE_OPTIONAL_HEADER32 has data directories after 96 bytes
    # IMAGE_OPTIONAL_HEADER64 has data directories after 112 bytes
    if magic == 0x10B:
        data_dir_off = optional_header_off + 96
    elif magic == 0x20B:
        data_dir_off = optional_header_off + 112
    else:
        raise ValueError(f"Unknown optional header magic 0x{magic:X}")

    # Import directory is entry index 1
    import_rva = _u32(data, data_dir_off + 8 * 1)
    _import_size = _u32(data, data_dir_off + 8 * 1 + 4)

    section_table_off = optional_header_off + size_of_optional_header
    sections = _parse_sections(data, section_table_off, number_of_sections)

    if import_rva == 0:
        return []

    import_off = _rva_to_offset(import_rva, sections)

    names: list[str] = []
    idx = 0
    while True:
        desc_off = import_off + idx * 20
        if desc_off + 20 > len(data):
            raise ValueError("Import descriptor table truncated")

        (
            orig_first_thunk,
            time_date_stamp,
            forwarder_chain,
            name_rva,
            first_thunk,
        ) = struct.unpack_from("<IIIII", data, desc_off)

        if (
            orig_first_thunk == 0
            and time_date_stamp == 0
            and forwarder_chain == 0
            and name_rva == 0
            and first_thunk == 0
        ):
            break

        name_off = _rva_to_offset(name_rva, sections)
        names.append(_read_cstring(data, name_off))

        idx += 1
        if idx > 2048:
            raise ValueError("Too many import descriptors (corrupt file?)")

    # Deduplicate while keeping order
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        key = n.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def try_load_dll(name: str) -> tuple[bool, str]:
    try:
        ctypes.WinDLL(name)
        return True, ""
    except OSError as e:
        return False, str(e)


def main() -> int:
    root = Path(__file__).resolve().parent
    pe_path = root / "Lib" / "site-packages" / "celux_cuda" / "_celux.pyd"

    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())
    print("Root:", root)
    print("Target:", pe_path)

    if not pe_path.exists():
        print("ERROR: _celux.pyd not found at expected path")
        return 2

    dll_dirs = [
        root,
        root / "ffmpeg",
        root / "Lib" / "site-packages" / "celux_cuda",
        root / "Lib" / "site-packages" / "nelux.libs",
        root / "Lib" / "site-packages" / "torch" / "lib",
    ]

    handles = []
    if hasattr(os, "add_dll_directory"):
        for d in dll_dirs:
            if d.exists():
                try:
                    handles.append(os.add_dll_directory(str(d)))
                    print(f"Added DLL dir: {d}")
                except Exception as e:
                    print(f"Failed to add DLL dir: {d} ({e})")
            else:
                print(f"Missing DLL dir: {d}")
    else:
        print("WARNING: os.add_dll_directory not available in this Python build")

    try:
        import torch

        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
    except Exception as e:
        print("torch import failed:", repr(e))

    try:
        names = get_imported_dll_names(pe_path)
    except Exception as e:
        print("ERROR: failed to parse PE imports:", repr(e))
        return 3

    if not names:
        print("No import-table DLLs found.")
        return 0

    print("\nDirect import-table DLLs:")
    for n in names:
        ok, err = try_load_dll(n)
        if ok:
            print(f"  OK   {n}")
        else:
            print(f"  FAIL {n} :: {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
