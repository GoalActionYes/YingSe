import importlib
import importlib.util
import json
import unittest
import numpy as np


class CategoryTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('vision_categories'))
        return importlib.import_module('vision_categories')

    def test_valid_categories_map_to_directions(self):
        api = self.api()
        region = dict(brightness='darker',temperature='cooler',saturation='lower',contrast='higher')
        plan = api.parse_categories(json.dumps(dict(person=region,background=region)))
        self.assertLess(plan[0,0],0)
        self.assertGreater(plan[0,1],0)
        self.assertLess(plan[0,2],0)
        self.assertLess(plan[0,4],0)
        np.testing.assert_array_equal(plan[0],plan[1])

    def test_same_is_zero(self):
        api = self.api()
        region = dict.fromkeys(('brightness','temperature','saturation','contrast'),'same')
        np.testing.assert_array_equal(api.parse_categories(json.dumps(dict(person=region,background=region))),np.zeros((2,6)))

    def test_ambiguous_or_unexpected_output_is_rejected(self):
        api = self.api()
        region = dict.fromkeys(('brightness','temperature','saturation','contrast'),'same')
        for raw in (json.dumps(dict(person=region,background=region,note='ok')),
                    json.dumps(dict(person=dict(region,brightness='warm'),background=region)),
                    'explanation '+json.dumps(dict(person=region,background=region)),
                    '{"person":{},"person":{},"background":{}}'):
            with self.assertRaises(ValueError): api.parse_categories(raw)


if __name__ == '__main__': unittest.main()
