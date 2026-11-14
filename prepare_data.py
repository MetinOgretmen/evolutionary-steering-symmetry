"""Restore saved controller weights and numerical outputs from the bundled archive."""

from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "reference_runs.zip"


def restore():
    pending = []
    unchanged = 0
    with ZipFile(ARCHIVE) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[0] != "results" or "\\" in entry.filename):
                raise ValueError(f"Invalid archive path: {entry.filename}")
            if entry.is_dir():
                continue
            destination = ROOT.joinpath(*path.parts)
            if not destination.resolve().is_relative_to(ROOT):
                raise ValueError(f"Archive path leaves the project: {entry.filename}")
            data = archive.read(entry)
            if destination.exists():
                if not destination.is_file() or sha256(destination.read_bytes()).digest() != sha256(data).digest():
                    raise FileExistsError(f"Existing file differs: {destination}")
                unchanged += 1
            else:
                pending.append((destination, data))
    for destination, data in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(f"Restored {len(pending)} files; {unchanged} identical files already present.")


if __name__ == "__main__":
    restore()
