import unittest
from build_natural_review import method_roles


class ReviewMethodRolesTests(unittest.TestCase):
    def manifest(self):
        return dict(before_control='Current App epoch8',source_sha256={'check_triplet_reserved.py':'592e1793a03ffae6c2b02aae00a0b5560e3f33d6bedf155a5fb16dc800be6245'})

    def test_before_is_app8_and_current_is_global(self):
        roles=method_roles(self.manifest())
        self.assertEqual(roles['before'],'prior_app8_epoch8')
        self.assertEqual(roles['current'],'legacy_global_pipeline')

    def test_unknown_producer_or_metadata_cannot_silently_reuse_roles(self):
        for manifest in [{},dict(before_control='Original author model'),dict(before_control='Current App epoch8',source_sha256={'check_triplet_reserved.py':'changed'})]:
            with self.assertRaises(ValueError):method_roles(manifest)
