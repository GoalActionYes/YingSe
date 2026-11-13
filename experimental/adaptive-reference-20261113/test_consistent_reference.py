"""Train-only pairing and balanced schedule for reference-consistency study."""
import unittest
from consistent_reference import choose_training_reference,use_synthetic_reference

class ConsistentReferenceTests(unittest.TestCase):
    def test_reference_is_other_training_content_and_order_independent(self):
        paths=['content_images/00/a.png','content_images/00/b.png','content_images/01/c.tif']
        for key in range(32):
            a=choose_training_reference(paths[0],paths,key)
            self.assertIn(a,paths[1:])
            self.assertEqual(a,choose_training_reference(paths[0],list(reversed(paths)),key))

    def test_changing_target_id_does_not_allow_self_reference(self):
        paths=['a.png','b.png','c.png']
        for target in paths:
            for key in range(12):self.assertNotEqual(target,choose_training_reference(target,paths,key))

    def test_schedule_keeps_each_target_and_style_balanced(self):
        for epoch in range(1,9):
            self.assertEqual(sum(use_synthetic_reference(style,epoch) for style in range(4)),2)
        for style in range(4):
            self.assertEqual(sum(use_synthetic_reference(style,epoch) for epoch in range(1,9)),4)
            for epoch in range(1,8):self.assertNotEqual(use_synthetic_reference(style,epoch),use_synthetic_reference(style,epoch+1))

    def test_missing_target_and_single_content_fail(self):
        for target,paths in [('a.png',['a.png']),('other.png',['a.png','b.png'])]:
            with self.assertRaises(ValueError):choose_training_reference(target,paths,1)
        for style,epoch in [(-1,1),(4,1),(0,0),(True,1),(0,True)]:
            with self.assertRaises(ValueError):use_synthetic_reference(style,epoch)

if __name__=='__main__':unittest.main()
