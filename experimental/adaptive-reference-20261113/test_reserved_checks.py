import hashlib
from pathlib import Path
import tempfile
import unittest
from reserved_checks import verify_frozen_predictions


class ReservedPredictionTests(unittest.TestCase):
    def test_incomplete_corrupt_and_failed_predictions_block_gt_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);cases=[]
            for i in range(32):
                path=directory/f'{i}.png';path.write_bytes(bytes([i]))
                record=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                cases.append(dict(index=i,outputs={name:dict(record) for name in ['input','candidate','current','before']}))
            verify_frozen_predictions(directory,cases)
            with self.assertRaises(ValueError):verify_frozen_predictions(directory,cases[:31])
            cases[0]['failure']={'message':'failed'}
            with self.assertRaises(ValueError):verify_frozen_predictions(directory,cases)
            del cases[0]['failure'];cases[0]['outputs'].pop('before')
            with self.assertRaises(ValueError):verify_frozen_predictions(directory,cases)
            cases[0]['outputs']['before']=dict(cases[0]['outputs']['input']);(directory/'0.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_frozen_predictions(directory,cases)


if __name__=='__main__':unittest.main()
