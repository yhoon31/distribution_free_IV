
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import zipfile

BASE = Path(__file__).resolve().parent


def restore_archive(name):
    archive = BASE/'archives'/f'{name}.zip'
    manifest_path = BASE/'archives'/f'{name}_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    digest = manifest['archive_sha256'].lower()
    target = Path(tempfile.gettempdir())/'iv_prediction_runtime'/f'{name}_{digest[:12]}'
    marker = target/'.verified_archive'
    if marker.exists() and marker.read_text() == digest:
        return target
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Archive checksum mismatch: {archive}')
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            destination = (target/member.filename.replace('\\', '/')).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError('Unsafe archive path')
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, destination.open('wb') as sink:
                    shutil.copyfileobj(source, sink)
    marker.write_text(digest)
    return target


def ensure_vendor():
    loose = BASE/'vendor'
    if (BASE/'archives'/'vendor_manifest.json').exists():
        return restore_archive('vendor')
    return loose


def ensure_tex_cache():
    loose = BASE/'latex_tools'/'cache'
    if (BASE/'archives'/'tex_cache_manifest.json').exists():
        return restore_archive('tex_cache')
    return loose
