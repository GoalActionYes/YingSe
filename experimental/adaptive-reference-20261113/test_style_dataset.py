import unittest
from style_dataset import split_groups


class StyleDatasetTests(unittest.TestCase):
    def test_stratified_identity_split_is_disjoint_and_deterministic(self):
        groups = {name: [{'id': group*1000+i} for i in range(100)]
                  for group, name in enumerate(('single', 'multiple', 'none'))}
        a = split_groups(groups)
        self.assertEqual(a, split_groups(groups))
        self.assertEqual({k: len(v) for k, v in a.items()}, {'train': 240, 'validation': 30, 'test': 30})
        ids = {k: {x['id'] for x in v} for k, v in a.items()}
        self.assertFalse(ids['train'] & ids['validation'] | ids['train'] & ids['test'] | ids['validation'] & ids['test'])

    def test_duplicate_identity_or_wrong_group_size_is_rejected(self):
        groups = {name: [{'id': i} for i in range(100)] for name in ('single', 'multiple', 'none')}
        with self.assertRaises(ValueError):
            split_groups(groups)
        with self.assertRaises(ValueError):
            split_groups({'single': []})


if __name__ == '__main__':
    unittest.main()
