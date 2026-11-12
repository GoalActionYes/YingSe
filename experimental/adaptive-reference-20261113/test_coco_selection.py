import unittest
from independent_coco import select_images,make_pairs


class SelectionChecks(unittest.TestCase):
    def test_selection_excludes_crowd_and_ineligible_license(self):
        images=[dict(id=i,width=640,height=480,license=4) for i in range(9)]
        images[0]['license']=6
        annotations=[dict(image_id=1,category_id=1,iscrowd=1,area=50000),
                     dict(image_id=2,category_id=1,iscrowd=0,area=50000),
                     dict(image_id=3,category_id=1,iscrowd=0,area=30000),
                     dict(image_id=3,category_id=1,iscrowd=0,area=30000)]
        for ann in annotations:ann['segmentation']=[[0,0,10,0,10,10]]
        result=select_images(images,annotations,count=1)
        self.assertEqual(result['single'][0]['id'],2)
        self.assertEqual(result['multiple'][0]['id'],3)
        self.assertEqual(result['none'][0]['id'],4)

    def test_pairing_is_disjoint_and_split_deterministic(self):
        groups={name:[dict(id=offset+i) for i in range(20)]
                for offset,name in [(0,'single'),(100,'multiple'),(200,'none')]}
        pairs=make_pairs(groups)
        self.assertEqual(len(pairs),30)
        ids=[p[key] for p in pairs for key in ('target','reference')]
        self.assertEqual(len(set(ids)),60)
        self.assertEqual(sum(p['split']=='development' for p in pairs),6)
        self.assertEqual(sum(p['split']=='evaluation' for p in pairs),24)


if __name__=='__main__':unittest.main()
