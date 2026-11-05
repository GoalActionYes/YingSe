import hashlib
import unittest
from pathlib import Path
import numpy as np
from reference_color_app.parameters import solve_parameters, _base_scalar, _mask_q, TABLE_SHA256
from test_matrices import packets

class ParameterTests(unittest.TestCase):
    def test_literal_layout_and_consumption(self):
        s,r=packets();s0=s.copy();r0=r.copy()
        p,d=solve_parameters(s,r)
        self.assertEqual(p.shape,(37,4));self.assertEqual(p.dtype,np.float32)
        np.testing.assert_array_equal(p[4],[1,3,2,4]);np.testing.assert_allclose(p[8],np.array([30,80,160,220])/255)
        np.testing.assert_allclose(p[9:12,:3],np.eye(3),atol=3e-6);np.testing.assert_array_equal(p[27,:3],[1,0,0])
        self.assertTrue(np.isnan(p[9:37,3]).all());self.assertTrue(d['consumed_values_finite'])
        self.assertEqual(d['internal_strength'],1);self.assertEqual(d['shader_layer_strengths'],[.8,1.])
        np.testing.assert_array_equal(s,s0);np.testing.assert_array_equal(r,r0)
    def test_without_uv_and_zero_strength(self):
        s,r=packets();r[:12]=0
        p,d=solve_parameters(s,r,0)
        self.assertTrue(d['without_uv']);self.assertTrue(np.isnan(p[3,2:]).all());self.assertTrue(np.isnan(p[4:]).all())
        np.testing.assert_array_equal(p[0,:2],[0,0]);np.testing.assert_array_equal(p[3,:2],[0,1])
    def test_ieee_zero_denominator(self):
        s,h,d=_base_scalar(0,0,0,0,False)
        self.assertEqual((s,h),(-1,-1));self.assertEqual(d['shadow_denominator'],0)
        self.assertEqual(_mask_q(0,np.float32(.1))[-1],2)
        self.assertEqual(_mask_q(0,np.float32(.15))[-1],np.float32(.2))
    def test_saturation_clamps_and_tables(self):
        s,r=packets();r[39:42]=[0,1,5]
        p,d=solve_parameters(s,r);np.testing.assert_allclose(p[36,:3],[.7,1/1.0001,1])
        path=Path(__file__).resolve().parents[2]/'reference_color_app/curve_tables.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "21e1ef57d474a2d177fb0bf83d6e8bc1118f59dd679a0e59505ae3192c481e55")
    def test_scalar_branches_and_translation_interleave(self):
        from reference_color_app.parameters import scalar_rules
        s,r=packets()
        for source,ref,branch in [([20,70,180],[100,100,220],'both_source_darker'),([100,120,220],[30,80,200],'both_source_brighter'),([20,70,220],[30,80,200],'opposite_or_zero_deltas')]:
            s[60:63]=source;r[54:57]=ref
            self.assertEqual(scalar_rules(s,r)[-1]['branch'],branch)
        r[14]=.16;r[23]=.04;r[32]=-.02
        p,d=solve_parameters(s,r)
        np.testing.assert_allclose(p[27:30,2],[.08,.04,-.02])
        np.testing.assert_allclose(p[18:21,:3],r[30:39].reshape(3,3))
        self.assertEqual(p[1,2],p[1,3])

    def test_invalid(self):
        s,r=packets()
        for strength in [-1,2,float('nan')]:
            with self.assertRaises(ValueError):solve_parameters(s,r,strength)
        with self.assertRaises(ValueError):solve_parameters(s[:-1],r)
