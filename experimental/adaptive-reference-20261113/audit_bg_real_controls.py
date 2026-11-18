"""Verify the fixed BG-REAL matched real-background metadata."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re


REVISION = 'aaba6cd1ced4ce3c846be9641fed092adb4919d1'
FILES = {
    'manifest_v2.csv': (12922417, 'aa75799c4c5e179639bd2d1d79b40cef5aea56df'),
    'attribution.csv': (1078451, '7bbb432adf887561f3ea14c3b3ca307e4853b3e0'),
    'recipients.csv': (405123, '56f444a7340d923f0e0c94f488796121a3a0397b'),
    'donors.csv': (165251, '1969f803a26cd56846da37a2da3f56a610c5e7f3'),
}
CC_BY = 'https://creativecommons.org/licenses/by/2.0/'
GEOMETRY = ('width','height','encoding','jpeg_quality')
MANIFEST_LFS = '773548c9aa03011f308059b87ac47fe72110783b8d70804dc6d4f837372cf5f4'


def verified_csv(folder, name):
    raw = (folder/name).read_bytes()
    size, oid = FILES[name]
    assert len(raw)==size
    sha256 = hashlib.sha256(raw).hexdigest()
    pointer = raw
    if name=='manifest_v2.csv':
        assert sha256 == MANIFEST_LFS
        pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n'.encode()
    assert hashlib.sha1(f'blob {len(pointer)}\0'.encode()+pointer).hexdigest()==oid
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    return rows, dict(bytes=size, blob_sha1=oid, sha256=sha256)


def item_path(path, prefix):
    key = PurePosixPath(path)
    assert not key.is_absolute() and '..' not in key.parts and key.as_posix().startswith(prefix)
    return key.as_posix()


def run(folder, out):
    data, files = {}, {}
    for name in FILES:
        data[name], files[name] = verified_csv(folder,name)
    recipients = {row['source_image_id']:row for row in data['recipients.csv']
                  if row['split']=='test_id' and row['subject_group']=='person'}
    attribution = {row['bgreal_id']:row for row in data['attribution.csv']}
    donors = {row['donor_id']:row for row in data['donors.csv']}
    assert len(attribution)==len(data['attribution.csv']) and len(donors)==len(data['donors.csv'])
    grouped = defaultdict(lambda:defaultdict(list))
    for row in data['manifest_v2.csv']:
        key = row['source_image_id']
        if key in recipients and row['split']=='test_id' and row['robustness']=='none':
            grouped[key][row['family']].append(row)
    rejected = Counter()
    paired, used_donors = {}, set()
    for key in sorted(recipients):
        families = grouped[key]
        if any(len(families[family])!=1 for family in ('C1','F1')):
            rejected['missing_or_duplicate']+=1
            continue
        empty,replaced = (families[family][0] for family in ('C1','F1'))
        donor = replaced['donor_id']
        if (empty['control_of']!='F1' or empty['label']!='0' or replaced['label']!='1'
            or empty['generator']!=replaced['generator'] or not empty['generator']
            or empty['mask_subject_path']!=replaced['mask_subject_path']
            or any(empty[field]!=replaced[field] for field in GEOMETRY)):
            rejected['protocol_mismatch']+=1
            continue
        if donor not in donors or any(attribution.get(value,{}).get('License')!=CC_BY
                                     or not attribution[value]['OriginalURL'] for value in (key,donor)):
            rejected['missing_source_attribution']+=1
            continue
        if donor==key or donor in used_donors:
            rejected['repeated_donor']+=1
            continue
        used_donors.add(donor)
        arms = {}
        for row in (empty,replaced):
            assert re.fullmatch('[a-f0-9]{64}',row['sha256'])
            arms[row['family']] = dict(path=item_path(row['image_path'],'v2/images/'),sha256=row['sha256'])
        paired[key] = dict(subject=recipients[key]['subject_class'],donor=donor,
                           donor_scene=donors[donor]['scene_category'],
                           subject_mask=item_path(replaced['mask_subject_path'],'v2/masks/subject/'),
                           shape=[int(replaced['width']),int(replaced['height'])],
                           encoding=replaced['encoding'],jpeg_quality=replaced['jpeg_quality'],
                           arms=arms)
    report = dict(revision=REVISION, files=files, data_rows={name:len(rows) for name,rows in data.items()},
                  person_recipients=len(recipients), valid_pairs=len(paired), rejected=rejected,
                  subject_breakdown=Counter(row['subject'] for row in paired.values()),
                  donor_scene_breakdown=Counter(row['donor_scene'] for row in paired.values()),
                  pairs=paired, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with out.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps({key:report[key] for key in ('person_recipients','valid_pairs','rejected','subject_breakdown')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    run(args.folder,args.out)
