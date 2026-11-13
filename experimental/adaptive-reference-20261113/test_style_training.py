import unittest
from pathlib import Path
from train_style_probe import load_images, fixed_validation_codes


class StyleTrainingTests(unittest.TestCase):
    def test_training_loader_rejects_final_test_before_file_read(self):
        with self.assertRaises(ValueError):
            load_images(Path('does-not-exist.json'), 'test', 'cpu')

    def test_validation_controls_are_fixed_and_include_identity_global_local(self):
        codes = fixed_validation_codes()
        self.assertTrue(bool((codes == fixed_validation_codes()).all()))
        self.assertTrue(bool((codes[:2] == 0).all()))
        self.assertTrue(bool((codes[2:5, 0] == codes[2:5, 1]).all()))
        self.assertTrue(bool((codes[5:, 0] != codes[5:, 1]).any()))


if __name__ == '__main__':
    unittest.main()
