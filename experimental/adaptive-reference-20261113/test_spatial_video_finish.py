import unittest
from types import SimpleNamespace
import numpy as np
from spatial_jobs import grade_video_frame
from reference_color_app.adjustments import normalize,transform


class SpatialVideoFinishTests(unittest.TestCase):
    def test_full_grade_then_strength_and_manual_adjustment_match_photo_policy(self):
        source=np.array([[[30,150,240],[190,20,110]]],np.uint8);graded=np.array([[[255,0,245],[0,220,150]]],np.uint8)
        strengths=[]
        def apply(frame,strength):strengths.append(strength);return graded.copy()
        engine=SimpleNamespace(prepare=lambda frame,ref:SimpleNamespace(apply=apply))
        settings=normalize();settings['basic']['contrast']=23
        result=grade_video_frame(engine,source,np.zeros_like(source),.5,settings)
        expected=np.rint(transform(source.astype(np.float32)/255*.5+graded.astype(np.float32)/255*.5,settings)*255).astype(np.uint8)
        np.testing.assert_array_equal(result,expected);self.assertEqual(strengths,[1])

    def test_zero_or_identical_reference_resets_history_and_keeps_manual_controls(self):
        source=np.full((2,3,3),80,np.uint8);settings=normalize();settings['basic']['brightness']=12
        def forbidden(*args):raise AssertionError('Identity should not invoke the model')
        engine=SimpleNamespace(prepare=forbidden);resets=[];temporal=SimpleNamespace(reset=lambda:resets.append(True))
        for strength,reference in [(0,np.zeros_like(source)),(1,source.copy())]:
            result=grade_video_frame(engine,source,reference,strength,settings,temporal)
            expected=np.rint(transform(source.astype(np.float32)/255,settings)*255).astype(np.uint8)
            np.testing.assert_array_equal(result,expected)
        self.assertEqual(len(resets),2)


if __name__=='__main__':unittest.main()
