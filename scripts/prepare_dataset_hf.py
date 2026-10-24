#!/usr/bin/env python3
"""Download the project's exact HF archive and unpack FF++, Celeb-DF, DFDC.

Never recompress media. Inspect ZIP member paths; fail rather than silently
leaving one required class empty. HF_TOKEN is read by huggingface_hub if needed.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / 'data/raw/Hybrid_Deepfake_Dataset'
SOURCES = {'ff++':'FF++', 'faceforensics++':'FF++', 'faceforensics':'FF++',
           'ffpp':'FF++', 'celeb-df':'Celeb-DF', 'celeb_df':'Celeb-DF',
           'celebdf':'Celeb-DF', 'dfdc':'DFDC'}
LABEL_ALIASES = {'real':'Real','realvideos':'Real','real_videos':'Real',
                 'fake':'Fake','fakevideos':'Fake','fake_videos':'Fake'}
MEDIA_EXTS = {'.mp4','.mov','.avi','.mkv','.m4v','.3gp','.webm'}


def dest_member(name):
    parts = list(Path(name.replace('\\','/')).parts)
    if '..' in parts or any(x.startswith('/') for x in parts):
        raise ValueError(f'Unsafe ZIP entry: {name}')
    for pos, part in enumerate(parts):
        key = part.casefold()
        ds = SOURCES.get(key)
        if ds is None:
            continue
        rest = parts[pos+1:]
        if not rest or parts[-1].endswith('/'):
            return None
        # Only recognize explicit Real/Fake directories, never infer labels from a filename.
        for j, p in enumerate(rest[:-1]):
            lab = LABEL_ALIASES.get(p.casefold())
            if lab:
                return Path(ds) / lab / Path(*rest[j+1:])
        # Sidecar files are only retained if they are under a known label directory.
    return None


def unpack(zippath, dest, inspect=False):
    counts = Counter()
    manifest = []
    skipped = Counter()
    with zipfile.ZipFile(zippath) as archive:
        # Avoid a costly full decompression of a multi-GB archive during --inspect.
        for item in archive.infolist():
            if item.is_dir(): continue
            rel = dest_member(item.filename)
            if rel is None:
                skipped['unrecognized_paths'] += 1
                continue
            if rel.suffix.lower() not in MEDIA_EXTS and rel.suffix.lower() not in {'.json','.csv','.txt'}:
                skipped['unrecognized_types'] += 1
                continue
            key = rel.parts[0] + '/' + rel.parts[1]
            if rel.suffix.lower() in MEDIA_EXTS: counts[key] += 1
            manifest.append((item, rel))
        if inspect:
            print('Recognized videos:', dict(counts))
            print('Skipped entries:', dict(skipped))
            print('Example member paths:', [x.filename for x,_ in manifest[:10]])
            return counts
        missing = [f'{d}/{c}' for d in ['FF++','Celeb-DF','DFDC'] for c in ['Real','Fake'] if counts[f'{d}/{c}']==0]
        if missing:
            examples = [x.filename for x in archive.infolist()[:30]]
            raise RuntimeError('Required classes missing from ZIP: '+', '.join(missing)+
                               '\nCheck true archive paths before extraction. First entries:\n'+'\n'.join(examples))
        dest.mkdir(parents=True, exist_ok=True)
        out_manifest=[]
        for n, (item, rel) in enumerate(manifest, 1):
            dst = dest / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.stat().st_size == item.file_size:
                state='existing'
            else:
                part = dst.with_name(dst.name + '.partial')
                with archive.open(item) as src, part.open('wb') as target:
                    shutil.copyfileobj(src, target, length=1024*1024)
                if part.stat().st_size != item.file_size:
                    raise IOError(f'Incomplete extraction: {dst}')
                part.replace(dst)
                state='extracted'
            out_manifest.append({'zip_member':item.filename,'destination':str(rel),
                                 'bytes':item.file_size,'status':state})
        # Save inside the DVC output folder, not outside it.
        with (dest/'hf_extraction_manifest.csv').open('w',newline='',encoding='utf8') as fh:
            writer=csv.DictWriter(fh,fieldnames=['zip_member','destination','bytes','status'])
            writer.writeheader();writer.writerows(out_manifest)
        (dest/'hf_extraction_summary.json').write_text(json.dumps({
            'counts':dict(counts),'skipped':dict(skipped),'archive':zippath.name,
            'files_processed':len(manifest)},indent=2),encoding='utf8')
        print('Extracted dataset/class counts:',dict(counts), flush=True)
        return counts


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-id',default='dinithiKP/Deepfakes')
    ap.add_argument('--archive',default='Deepfakes.zip')
    ap.add_argument('--revision',default='main')
    ap.add_argument('--local-zip',help='Use already downloaded ZIP for offline runs')
    ap.add_argument('--dest',default=str(DEFAULT_DEST))
    ap.add_argument('--inspect',action='store_true',help='List recognized classes without extracting')
    args=ap.parse_args()
    if args.local_zip:
        zippath=Path(args.local_zip).resolve()
    else:
        try: from huggingface_hub import hf_hub_download
        except ImportError as err: raise SystemExit('Install huggingface_hub: pip install huggingface_hub') from err
        zippath=Path(hf_hub_download(repo_id=args.repo_id, filename=args.archive,
                                     repo_type='dataset', revision=args.revision))
    if not zippath.is_file(): raise FileNotFoundError(zippath)
    print('Archive:',zippath,'size_bytes:',zippath.stat().st_size,flush=True)
    unpack(zippath, Path(args.dest).resolve(), inspect=args.inspect)

if __name__=='__main__': main()
