import unittest,json,ast
from pathlib import Path
import numpy as np


class GateSerializationTests(unittest.TestCase):
    def test_producer_added_reference_gate_serializes_all_outcomes(self):
        tree=ast.parse(Path(__file__).with_name('train_petal_condition.py').read_text(encoding='utf-8'))
        nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='gates' and isinstance(t.slice,ast.Constant) and t.slice.value=='added_reference_passed' for t in n.targets)]
        self.assertEqual(len(nodes),1)
        code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'real-producer-gate','exec')
        for count,after,expected in [(32,9.,True),(16,9.,False),(32,11.,False)]:
            env=dict(np=np,gates={},added_a=[dict(index=i,mae=10.) for i in range(count)],added_b={i:after for i in range(count)})
            exec(code,env)
            saved=json.loads(json.dumps(env['gates']))
            self.assertIs(saved['added_reference_passed'],expected)
