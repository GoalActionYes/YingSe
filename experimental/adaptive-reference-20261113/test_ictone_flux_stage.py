"""Check the real full architecture's empty disk staging before payload loads."""
import json
import tempfile
import unittest
from pathlib import Path

from ictone_flux_stage import build_empty_flux_groups

ROOT = Path(__file__).resolve().parents[2]


class StageTest(unittest.TestCase):
    def test_full_meta_architecture_and_source_coverage(self):
        metadata = json.loads((ROOT / 'outputs/adaptive-reference-20261113/ictone-quantized-metadata-v1/results.json').read_text())
        config = json.loads((ROOT / '.tools/ICTone-quantized-audit/models/components/transformer/config.json').read_text())
        # Files stay in the research output tree, not system temp or user data.
        parent = ROOT / 'outputs/adaptive-reference-20261113'
        with tempfile.TemporaryDirectory(prefix='ictone-empty-stage-test-', dir=parent) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(parent.resolve()))
            model, groups, identities = build_empty_flux_groups(metadata, config, Path(directory))
            self.assertEqual(len(groups), 58)
            self.assertEqual(len(identities), 1846)
            covered = []
            for group, names, inputs in groups:
                self.assertTrue(Path(group.safetensors_file_path).is_file())
                self.assertEqual(len(names), len(group.tensor_to_key))
                covered.extend(inputs)
                self.assertTrue(all(p.untyped_storage().nbytes() == 0 for p in group.tensor_to_key))
            self.assertEqual(sorted(covered), sorted(x['name'] for x in metadata['parsed']['fill']['tensors']))
            self.assertTrue(all(not p.is_meta and not p.requires_grad for p in model.parameters()))
            self.assertEqual(set(identities), {name for name, _ in model.named_parameters()})


if __name__ == '__main__':
    unittest.main()
