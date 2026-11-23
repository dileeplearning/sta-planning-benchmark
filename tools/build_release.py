"""Build a deterministic ZIP containing only this focused release package."""
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'sta-planning-benchmark-v0.1.0'
EXCLUDE = {'.git', '.venv', '__pycache__', 'scratch', 'dist', '.cache'}


def main():
    archive = ROOT.parent / f'{NAME}.zip'
    files = [p for p in ROOT.rglob('*') if p.is_file()
             and not (set(p.relative_to(ROOT).parts) & EXCLUDE)
             and p.name != '.DS_Store' and p.suffix != '.pyc']
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(files):
            item = zipfile.ZipInfo(f'{NAME}/{path.relative_to(ROOT).as_posix()}', (2026, 11, 23, 0, 0, 0))
            item.external_attr = 0o100644 << 16
            item.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(item, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n')
    print(f'{archive.name}: {len(files)} files, {archive.stat().st_size:,} bytes\nSHA-256: {digest}')


if __name__ == '__main__':
    main()
