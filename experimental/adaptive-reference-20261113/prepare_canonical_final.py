"""Verify and extract only the two official weight files from a local public ZIP."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from independent_coco import ROOT,digest

ARCHIVE_SHA='ab460d12e0f826a507d54fc55ac6842ae935b1e83916d7259975e068c886cc51'
SSL_SHA='5630c6dd5ceefa0d36d69ad298a5b87f1260dd02620a60af84cb4e259b4ad362'
FILES={'E2E.pth':(20914369,'916f7ad5028d3fef51cf915bb51febc9508e7a378c0babe9ff3d3992b8c594f7'),
       'SSL_updated_251111.pth':(20882298,SSL_SHA)}


def main():
    import hashlib
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,default=ROOT/'.downloads/canoncgt-final-package.zip');args=parser.parse_args()
    if args.archive.stat().st_size!=37783740 or digest(args.archive)!=ARCHIVE_SHA: raise ValueError('Official package transfer differs')
    out=ROOT/'.tools/CanonCGT-final'; out.mkdir(exist_ok=True)
    entries=[]
    with zipfile.ZipFile(args.archive) as archive:
        if set(archive.namelist())!=set(FILES) or len(archive.infolist())!=len(FILES): raise ValueError('Unexpected archive entries')
        for name,(size,expected) in FILES.items():
            raw=archive.read(name)
            if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('Weight checksum mismatch')
            path=out/name
            if path.exists() and path.read_bytes()!=raw: raise ValueError('Existing weight differs')
            if not path.exists(): path.write_bytes(raw)
            entries.append(dict(name=name,bytes=size,sha256=expected))
    license_path=ROOT/'.tools/CanonCGT/LICENSE'
    shutil.copyfile(license_path,out/'LICENSE')
    manifest=dict(source_url='https://drive.google.com/file/d/1SqzCXjdJ95TAhDYY9Z4TaQPuoqlEyfkT/view',
        archive_bytes=37783740,archive_sha256=ARCHIVE_SHA,files=entries,license_sha256=digest(out/'LICENSE'),
        source='Official README public download; connector obtained raw ZIP; hashes recorded locally, not author-signed')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__': main()
