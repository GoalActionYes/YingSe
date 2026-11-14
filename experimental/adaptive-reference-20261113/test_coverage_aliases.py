import importlib
import unittest


class CoverageAliasTests(unittest.TestCase):
    def api(self):return importlib.import_module('coverage_aliases')
    def test_transitive_closure_excludes_entire_component_only_when_heldout(self):
        got=self.api().heldout_alias_closure([('a','b'),('c','b'),('x','y')],{'c','z'})
        self.assertEqual(got,{'a','b','c','z'})
    def test_edge_order_and_duplicate_edges_do_not_change_closure(self):
        api=self.api();edges=[('a','b'),('b','c'),('d','e')]
        self.assertEqual(api.heldout_alias_closure(edges,{'a'}),api.heldout_alias_closure(list(reversed(edges))+[('b','a')],{'a'}))
    def test_bad_alias_ids_and_edges_stop(self):
        api=self.api()
        for edges in [[('a',None)],[('a','a')],[('a','')],[('a','b','c')]]:
            with self.assertRaises(ValueError):api.heldout_alias_closure(edges,{'z'})


if __name__=='__main__':unittest.main()
