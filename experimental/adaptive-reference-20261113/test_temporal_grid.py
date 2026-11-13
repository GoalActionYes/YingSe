import unittest
import numpy as np
import torch
from spatial_render import PreparedSpatial
from temporal_grid import TemporalGrid


class TemporalGridTests(unittest.TestCase):
    def prepared(self,engine,value):return PreparedSpatial(engine,torch.full((1,13,8,16,16),float(value)),{})

    def test_smooths_coefficients_without_modifying_input_or_previous_results(self):
        state=TemporalGrid();engine=object();frame=np.zeros((32,32,3),np.uint8)
        first=state.smooth(self.prepared(engine,0),frame)
        current=self.prepared(engine,1);second=state.smooth(current,frame)
        torch.testing.assert_close(second.grid,torch.full_like(second.grid,.2))
        torch.testing.assert_close(current.grid,torch.ones_like(current.grid))
        torch.testing.assert_close(first.grid,torch.zeros_like(first.grid))
        with torch.inference_mode():second.grid.fill_(100)
        third=state.smooth(self.prepared(engine,1),frame)
        torch.testing.assert_close(third.grid,torch.full_like(third.grid,.36))

    def test_cut_engine_change_and_explicit_reset_use_current_grid_immediately(self):
        state=TemporalGrid();engine=object();black=np.zeros((32,32,3),np.uint8);white=np.full_like(black,255)
        state.smooth(self.prepared(engine,0),black)
        cut=state.smooth(self.prepared(engine,1),white)
        self.assertEqual(cut.diagnostics['temporal_reset_reason'],'scene_cut');self.assertTrue(bool((cut.grid==1).all()))
        changed=state.smooth(self.prepared(object(),3),white)
        self.assertEqual(changed.diagnostics['temporal_reset_reason'],'engine_changed');self.assertTrue(bool((changed.grid==3).all()))
        resized=state.smooth(self.prepared(changed.engine,4),white[:16])
        self.assertEqual(resized.diagnostics['temporal_reset_reason'],'dimensions_changed');self.assertTrue(bool((resized.grid==4).all()))
        state.reset();fresh=state.smooth(self.prepared(engine,5),black)
        self.assertEqual(fresh.diagnostics['temporal_reset_reason'],'first_frame');self.assertTrue(bool((fresh.grid==5).all()))

    def test_invalid_parameters_and_nonfinite_grid_fail(self):
        for kwargs in [dict(alpha=0),dict(alpha=True),dict(alpha=float('nan')),dict(cut_threshold=-1)]:
            with self.assertRaises(ValueError):TemporalGrid(**kwargs)
        p=self.prepared(object(),float('nan'))
        with self.assertRaises(ValueError):TemporalGrid().smooth(p,np.zeros((32,32,3),np.uint8))


if __name__=='__main__':unittest.main()
