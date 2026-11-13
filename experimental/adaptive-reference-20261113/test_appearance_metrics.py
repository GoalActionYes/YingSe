import unittest
from appearance_metrics import assess_appearance


def good_rows():
    rows = []
    for pair in range(15):
        group = ('single', 'multiple', 'none')[pair//5]
        for style in range(10):
            same = style < 4
            local = group != 'none' and style >= 5
            rows.append({'target': pair, 'reference': pair+100, 'style_index': style, 'group': group,
                         'same_style': same, 'local_control': local,
                         'input_mae': 0. if same else 20., 'output_mae': .5 if same else 10.,
                         'input_relation_error': 10. if local else None, 'output_relation_error': 5. if local else None})
    return rows


class AppearanceMetricTests(unittest.TestCase):
    def test_all_acceptance_conditions_pass(self):
        result = assess_appearance(good_rows())
        self.assertTrue(result['passed'])
        self.assertEqual(result['local_count'], 50)
        self.assertEqual(result['same_style_count'], 60)

    def test_false_edits_cannot_be_hidden_by_other_gains(self):
        rows = good_rows()
        for row in rows:
            if row['same_style']:
                row['output_mae'] = 4.
        self.assertFalse(assess_appearance(rows)['passed'])

    def test_local_relationships_are_required(self):
        rows = good_rows()
        for row in rows:
            if row['local_control']:
                row['output_relation_error'] = 11.
        self.assertFalse(assess_appearance(rows)['passed'])

    def test_partial_or_duplicate_results_fail(self):
        self.assertFalse(assess_appearance(good_rows()[:-1])['passed'])
        rows = good_rows()
        rows[-1] = rows[0].copy()
        self.assertFalse(assess_appearance(rows)['passed'])


if __name__ == '__main__':
    unittest.main()
