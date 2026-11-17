"""Screen integrity-checked PSBench instructions for comparable local grading tasks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

REVISION = '929d0710eb7c86a988f8a69a74619052b60ac293'
COLOR = re.compile(r'colou?r|hue|saturat|monochrom|gr[ae]yscale|black and white|tint|\bb&w\b|colour balance', re.I)
TONE = re.compile(r'contrast|brightness|exposure|highlights?|shadows?|lighten|darken|\bcurves?\b|\blevels\b', re.I)
REGION = re.compile(r'background|foreground|subject|skin|faces?|\bsky\b|\beyes?\b|wall|shadows?|highlights?|\bleft\b|\bright\b|\btop\b|\bbottom\b|center|centre|\bmask\b|selected|specific area|surround|only|except', re.I)


def canonical(instruction):
    return ' '.join(re.findall(r'[a-z0-9]+', instruction.lower()))


def run(folder, out):
    manifest_path = folder / 'manifest.json'
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    assert manifest['revision'] == REVISION and not manifest['errors']
    expected = [f'{n:03d}' for n in range(1, 601)]
    assert [row['id'] for row in manifest['files']] == expected
    matches = defaultdict(list)
    types = Counter()
    pool = []
    review_lines = []
    for row in manifest['files']:
        data = (folder / f"{row['id']}.json").read_bytes()
        assert len(data) == row['bytes'] <= 4096 and hashlib.sha256(data).hexdigest() == row['sha256']
        task = json.loads(data)
        assert task['id'] == row['id']
        text = task['instruction']
        if not ((COLOR.search(text) or TONE.search(text)) and REGION.search(text)):
            continue
        matches[canonical(text)].append(task['id'])
        types[task['type']] += 1
        pool.append(task['id'])
        review_lines.append(f"{task['id']} {task['type']} {text}")
    report = dict(revision=REVISION, reviewed_metadata=len(expected),
                  metadata_bytes=sum(row['bytes'] for row in manifest['files']),
                  manifest_sha256=hashlib.sha256(manifest_data).hexdigest(), local_color_keyword_matches=pool,
                  exact_normalized_duplicate_groups=[ids for ids in matches.values() if len(ids) > 1],
                  matched_task_types=dict(types), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256(Path(__file__).with_name('psbench-shared-local-intent-protocol.md').read_bytes()).hexdigest())
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    out.with_suffix('.review.txt').write_text('\n'.join(review_lines) + '\n', encoding='utf-8')
    print('Keyword candidates:', len(pool), 'repeated exact instructions:', len(report['exact_normalized_duplicate_groups']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.folder, args.out)
