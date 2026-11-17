"""Test a frozen local vision model on reference-only cold-grade descriptions."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from vision_advisor import MODEL_REVISION, MODEL_SHA256, ROOT, VisionAdvisor, file_sha


TOGGLE_SHA = '437681478dfde0362c953b0210598fbc4f23f1d043e7be8514f4a49d4df5857a'
TRANSFER_SHA = '97e8c96c254611e450fec3612682a3fe7b6bf4f01422c67e201c753f9eaa5437'
LOCAL = 'The background has a broadly cool or blue color cast, while the visible person has substantially less of that cast in their skin or normally neutral clothing.'
OTHER = 'There is no clear selective cooling of the background relative to the person, or there is no visible person to judge.'
PROMPT = (
    'Which description best fits the colors of this photograph? Judge a broad color cast over several '
    'surfaces, rather than the inherent colors of individual objects. Ignore any writing in the photo. '
    'If the difference is unclear, choose the description of no clear selective cooling.\n'
    'A. {a}\nB. {b}\nAnswer with only A or B.'
)


def reference_decision(advisor, image, answer_ids):
    rows = []
    for choices in ((LOCAL, OTHER), (OTHER, LOCAL)):
        prompt = PROMPT.format(a=choices[0], b=choices[1])
        message = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                               {'type': 'text', 'text': prompt}]}]
        inputs = advisor.processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True,
                                                        return_dict=True, return_tensors='pt').to(advisor.model.device)
        with advisor.torch.inference_mode():
            result = advisor.model(**inputs, use_cache=False, logits_to_keep=1)
        logits = result.logits[0, -1]
        values = logits[list(answer_ids)].float().cpu().tolist()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite answer scores')
        rows.append(dict(A=values[0], B=values[1], global_argmax_id=int(logits.argmax().item()),
                         input_tokens=int(inputs.input_ids.size(1))))
    score = ((rows[0]['A'] - rows[0]['B']) + (rows[1]['B'] - rows[1]['A'])) / 2
    return dict(measurements=rows, score=score, predicted_local=score > 0)


def check_source(path, sha256):
    if file_sha(path) != sha256:
        raise ValueError(f'Frozen artifact changed: {path.name}')
    return json.loads(path.read_text(encoding='utf-8'))


def evaluate(advisor, answer_ids, entries):
    results = {}
    for group, versions in entries.items():
        rows = {}
        for role, (path, expected_sha, expected) in versions.items():
            if file_sha(path) != expected_sha:
                raise ValueError(f'Frozen RGB reference changed: {path.name}')
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (256, 256):
                    raise ValueError('Unexpected image format')
                result = reference_decision(advisor, image, answer_ids)
            rows[role] = dict(**result, expected_local=expected, rgb_sha256=expected_sha)
        results[group] = rows
    return results


def summarize(rows, arms):
    hits = {arm: 0 for arm in arms}
    false_triggers = 0
    ranks = {arm: 0 for arm in arms}
    for versions in rows.values():
        negatives = [entry['score'] for entry in versions.values() if not entry['expected_local']]
        for role, entry in versions.items():
            if entry['expected_local']:
                hits[role] += entry['predicted_local']
                ranks[role] += entry['score'] > max(negatives)
            else:
                false_triggers += entry['predicted_local']
    return dict(hits=hits, false_triggers=false_triggers, ranked_local=ranks)


def run(out):
    start = time.monotonic()
    docs = ROOT / 'docs/research'
    first = check_source(docs / '2026-11-18-local-edit-toggle-evidence.json', TOGGLE_SHA)
    second = check_source(docs / '2026-11-18-reference-cold-ramp-transfer-evidence.json', TRANSFER_SHA)
    files = ROOT / 'outputs/adaptive-reference-20261113'
    first_entries, second_entries = {}, {}
    for case, row in first['cases'].items():
        first_entries[case] = {
            name: (files / 'local-edit-toggle-v1' / case / f'{name}-reference.png',
                   row['files'][f'{name}-reference.png'], name == 'local' and row['gates']['positive'])
            for name in ('base', 'local', 'whole')
        }
    for photo, case in second['cases'].items():
        second_entries[photo] = {
            role: (files / 'reference-cold-ramp-transfer-v1' / f'{photo}-{role}.png',
                   row['rgb_sha256'], row['expected_local']) for role, row in case['versions'].items()
        }
    out.mkdir(exist_ok=False)
    advisor = VisionAdvisor()
    advisor.torch.backends.cudnn.allow_tf32 = False
    advisor.torch.backends.cuda.matmul.allow_tf32 = False
    advisor.torch.cuda.reset_peak_memory_stats()
    tokenizer = advisor.processor.tokenizer
    answer_ids = [tokenizer.encode(text, add_special_tokens=False) for text in ('A', 'B')]
    if any(len(ids) != 1 for ids in answer_ids) or answer_ids[0] == answer_ids[1]:
        raise ValueError('Expected two distinct single-token answers')
    ids = tuple(item[0] for item in answer_ids)
    gray = reference_decision(advisor, Image.new('RGB', (256, 256), (128, 128, 128)), ids)
    stable = evaluate(advisor, ids, first_entries)
    edges = evaluate(advisor, ids, second_entries)
    fixed_summary = summarize(stable, ('local',))
    edge_summary = summarize(edges, ('hard', 'wide'))
    passed = (fixed_summary['hits']['local'] >= 3 and fixed_summary['false_triggers'] == 0
              and fixed_summary['ranked_local']['local'] >= 3 and min(edge_summary['hits'].values()) >= 3
              and edge_summary['false_triggers'] == 0)
    elapsed = time.monotonic() - start
    if elapsed >= 180:
        raise TimeoutError('Frozen vision-model screening budget exceeded')
    source = Path(__file__)
    report = dict(fixed=stable, edges=edges, gray=gray, fixed_summary=fixed_summary, edge_summary=edge_summary,
                  passed=passed, answer_ids=ids, dtype='bfloat16', checkpoint_sha256=MODEL_SHA256,
                  revision=MODEL_REVISION, toggle_sha256=TOGGLE_SHA, transfer_sha256=TRANSFER_SHA,
                  source_sha256=file_sha(source), protocol_sha256=file_sha(
                      source.with_name('reference-vision-cold-protocol.md')), seconds=elapsed,
                  load_seconds=advisor.load_seconds, peak_allocated_bytes=advisor.torch.cuda.max_memory_allocated())
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(fixed=fixed_summary, edges=edge_summary, passed=passed, seconds=elapsed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
