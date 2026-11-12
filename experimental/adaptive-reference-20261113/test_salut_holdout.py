import copy
import unittest
from salut_holdout import assess, validate_jobs


def fixture():
    rows = []
    profiles = ('identity', 'global_contrast', 'global_desaturate', 'region_color_luminance', 'region_tone_saturation')
    for pair in range(24):
        group = ('single', 'multiple', 'none')[pair//8]
        for profile in profiles:
            relation = None if group == 'none' else 10.
            rows.append({'target': pair, 'reference': pair+100, 'profile': profile, 'group': group,
                         'baseline': {'mae': 20., 'relation_error': relation},
                         'salut_inverse': {'mae': 15., 'relation_error': None if relation is None else 7.}})
    return rows


class HoldoutGateTests(unittest.TestCase):
    def test_complete_good_result_passes(self):
        result = assess(fixture())
        self.assertTrue(result['passed'])
        self.assertEqual(result['regional_count'], 32)

    def test_missing_or_duplicate_cases_cannot_pass(self):
        rows = fixture()
        self.assertFalse(assess(rows[:-1])['passed'])
        rows[-1] = copy.deepcopy(rows[0])
        self.assertFalse(assess(rows)['passed'])

    def test_large_gains_cannot_hide_severe_regressions(self):
        rows = fixture()
        for row in rows[:13]:
            row['salut_inverse']['mae'] = 30.
        result = assess(rows)
        self.assertFalse(result['passed'])
        self.assertEqual(result['severe_regressions'], 13)

    def test_whole_image_improvement_cannot_replace_region_relationship_gain(self):
        rows = fixture()
        for row in rows:
            if row['salut_inverse']['relation_error'] is not None:
                row['salut_inverse']['relation_error'] = 11.
        self.assertFalse(assess(rows)['passed'])

    def test_empty_or_failed_runs_cannot_pass(self):
        self.assertFalse(assess([])['passed'])
        rows = fixture()
        rows[0] = {'error': 'inference failure'}
        self.assertFalse(assess(rows)['passed'])

    def test_development_pair_cannot_be_substituted_into_holdout(self):
        rows = fixture()
        pairs = [{k: row[k] for k in ('target', 'reference', 'group')} | {'split': 'evaluation'} for row in rows[::5]]
        pairs.append({'target': 999, 'reference': 1999, 'group': 'single', 'split': 'development'})
        jobs = []
        for row in rows:
            jobs.append({k: row[k] for k in ('target', 'reference', 'group', 'profile')} | {'split': 'evaluation',
                         'reference_file': f'evaluation/references/{row["reference"]}-{row["profile"]}.png',
                         'truth_file': f'evaluation/truth/{row["target"]}-{row["profile"]}.png'})
        validate_jobs({'split': 'evaluation', 'jobs': jobs}, {'pairs': pairs})
        for row in jobs[:5]:
            row['target'], row['reference'] = 999, 1999
            row['reference_file'] = f'evaluation/references/1999-{row["profile"]}.png'
            row['truth_file'] = f'evaluation/truth/999-{row["profile"]}.png'
        with self.assertRaises(ValueError):
            validate_jobs({'split': 'evaluation', 'jobs': jobs}, {'pairs': pairs})


if __name__ == '__main__':
    unittest.main()
