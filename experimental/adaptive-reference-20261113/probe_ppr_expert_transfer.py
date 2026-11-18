"""Screen expert-conditioned editing across the fixed PPR development scenes."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from diagnose_ppr_spatial_information import basis, metrics
from independent_coco import digest
from spatial_render import SpatialEngine, finish_spatial


SCENES = ('4_2', '7_4', '15_3', '19_1')
EXPERTS = ('a', 'b', 'c')
INPUT_SHA = '61e88e0dddeb88202313917c55d4a27d745c9c6a67e054c12b5a2b1d651813f6'


def descriptor(image):
    return np.quantile(image.reshape(-1, 3), (.25, .50, .75), axis=0).ravel()


def calibrate(data, training):
    features = np.concatenate([basis(data[f'{scene}_source'], 3) for scene in training])
    gram = features.T @ features / len(features)
    penalty = np.eye(gram.shape[0])*1e-6
    penalty[0, 0] = 0
    centers, coefficients = [], []
    for expert in EXPERTS:
        grades = np.concatenate([data[f'{scene}_{expert}'].reshape(-1, 3) for scene in training])
        right = features.T @ (grades.astype(float)/255) / len(features)
        coefficients.append(np.linalg.solve(gram + penalty, right))
        centers.append(np.mean([descriptor(data[f'{scene}_{expert}']) for scene in training], axis=0))
    return np.array(centers), np.array(coefficients)


def improves(scores, app, base):
    decrease = scores[base]['all_mae'] - scores[app]['all_mae']
    return decrease >= .25 and decrease >= .05*scores[base]['all_mae'] and all(
        scores[app][region] <= scores[base][region] + .1 for region in ('human_mae', 'background_mae'))


def run(inputs, out):
    assert digest(inputs) == INPUT_SHA
    engine = SpatialEngine('multiref-v1')
    out.mkdir()
    cases = {}
    with np.load(inputs, allow_pickle=False) as data:
        for target_id, reference_id in zip(SCENES, SCENES[1:]+SCENES[:1]):
            training = [scene for scene in SCENES if scene not in (target_id, reference_id)]
            centers, coeff = calibrate(data, training)
            target = data[f'{target_id}_source']
            grid = basis(target, 3)

            def render(weights):
                graded = np.rint(np.clip(grid @ weights, 0, 1)*255).astype(np.uint8).reshape(target.shape)
                return finish_spatial(target, graded, .8)

            generic = render(coeff.mean(axis=0))
            per_expert = [render(weights) for weights in coeff]
            records, outputs = {}, {}
            for index, expert in enumerate(EXPERTS):
                reference = data[f'{reference_id}_{expert}']
                guess = int(np.argmin(np.linalg.norm(centers - descriptor(reference), axis=1)))
                baseline = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
                output = dict(generic=generic, oracle=per_expert[index], candidate=per_expert[guess],
                              current13=baseline)
                outputs[expert] = output
                files = {}
                for method, photo in output.items():
                    path = out / f'{target_id}-{expert}-{method}.png'
                    Image.fromarray(photo).save(path)
                    files[method] = digest(path)
                records[expert] = dict(guess=EXPERTS[guess], output_sha256=files)

            alpha = data[f'{target_id}_alpha']
            desired = {expert: finish_spatial(target, data[f'{target_id}_{expert}'], .8) for expert in EXPERTS}
            for expert, row in records.items():
                row['scores'] = {name: metrics(image, desired[expert], alpha)
                                 for name, image in outputs[expert].items()}
                row['candidate_error_by_goal'] = {goal: metrics(outputs[expert]['candidate'], desired[goal], alpha)
                                                  ['all_mae'] for goal in EXPERTS}
            means = {method: {score: float(np.mean([row['scores'][method][score] for row in records.values()]))
                             for score in ('all_mae', 'human_mae', 'background_mae')}
                     for method in outputs['a']}
            classified = sum(expert == row['guess'] for expert, row in records.items())
            oracle_pass = improves(means, 'oracle', 'generic')
            candidate_pass = classified >= 2 and all(improves(means, 'candidate', base)
                                                     for base in ('generic', 'current13'))
            cases[target_id] = dict(reference=reference_id, calibration=training,
                                    classified=classified, mean_scores=means, records=records,
                                    oracle_pass=bool(oracle_pass), candidate_pass=bool(candidate_pass))

    oracle_folds = sum(case['oracle_pass'] for case in cases.values())
    candidate_folds = sum(case['candidate_pass'] for case in cases.values())
    result = dict(inputs_sha256=INPUT_SHA, checkpoint_sha256=engine.checkpoint_sha,
                  source_sha256=digest(Path(__file__)), cases=cases, oracle_folds=oracle_folds,
                  candidate_folds=candidate_folds, proceeded=oracle_folds >= 3 and candidate_folds >= 3)
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(oracle_folds=oracle_folds, candidate_folds=candidate_folds,
                          classified=[case['classified'] for case in cases.values()])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.inputs, args.out)
