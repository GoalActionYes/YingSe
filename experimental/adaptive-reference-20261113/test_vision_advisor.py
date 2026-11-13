import importlib
import importlib.util
import unittest

import numpy as np


class VisionAdvisorTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('vision_advisor'), 'Advisor implementation is missing')
        return importlib.import_module('vision_advisor')

    def test_strict_plan_and_whole_json_fence(self):
        api = self.api()
        text = '{"person":[1,2,3,4,5,6],"background":[0,0,0,0,0,0]}'
        np.testing.assert_array_equal(api.parse_plan(text), [[1,2,3,4,5,6], [0]*6])
        np.testing.assert_array_equal(api.parse_plan('```json\n'+text+'\n```'), api.parse_plan(text))

    def test_invalid_output_is_rejected_not_silently_clamped(self):
        api = self.api()
        for text in ('{"person":[true,0,0,0,0,0],"background":[0,0,0,0,0,0]}',
                     '{"person":[NaN,0,0,0,0,0],"background":[0,0,0,0,0,0]}',
                     '{"person":[101,0,0,0,0,0],"background":[0,0,0,0,0,0]}',
                     '{"person":[0,0,0,0,0,0],"person":[0,0,0,0,0,0],"background":[0,0,0,0,0,0]}',
                     '{"person":[0,0,0,0,0,0],"background":[0,0,0,0,0,0],"note":"ok"}',
                     'Here is JSON: {"person":[0,0,0,0,0,0],"background":[0,0,0,0,0,0]}',
                     '{"person":[0,0],"background":[0,0,0,0,0,0]}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                api.parse_plan(text)

    def test_zero_and_strength_zero_are_exact_for_every_code_value(self):
        api = self.api()
        image = np.arange(256, dtype=np.uint8).reshape(16,16,1).repeat(3,axis=2)
        mask = np.linspace(0,1,256,dtype=np.float32).reshape(16,16)
        np.testing.assert_array_equal(api.render_plan(image, mask, np.zeros((2,6))), image)
        np.testing.assert_array_equal(api.render_plan(image, mask, np.ones((2,6))*100, 0), image)

    def test_oversized_integer_is_recordable_invalid_reply(self):
        api = self.api()
        import json
        raw = json.dumps({'person':[10**50,0,0,0,0,0],'background':[0]*6})
        with self.assertRaises(ValueError):
            api.parse_plan(raw)

    def test_regions_can_treat_identical_rgb_differently_and_feather(self):
        api = self.api()
        image = np.full((1,3,3),128,dtype=np.uint8)
        mask = np.array([[0,.5,1]],dtype=np.float32)
        plan = np.zeros((2,6)); plan[0,0] = 60; plan[1,1] = 60
        output = api.render_plan(image, mask, plan)
        self.assertGreater(output[0,2,0],output[0,2,2])
        self.assertGreater(output[0,0,2],output[0,0,0])
        self.assertLess(output[0,1,0],output[0,2,0])
        self.assertGreater(output[0,1,0],output[0,0,0])

    def test_invalid_arrays_and_strength_are_rejected(self):
        api = self.api()
        rgb = np.full((2,3,3),128,dtype=np.uint8)
        for mask, plan, strength in ((np.ones((2,2)),np.zeros((2,6)),1),
                                     (np.full((2,3),np.nan),np.zeros((2,6)),1),
                                     (np.ones((2,3)),np.full((2,6),101),1),
                                     (np.ones((2,3)),np.zeros((2,6)),1.01)):
            with self.assertRaises(ValueError): api.render_plan(rgb,mask,plan,strength)


if __name__ == '__main__':
    unittest.main()
