import importlib
import importlib.util
import unittest
import numpy as np
import torch


class FeatureAssayTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('reference_feature_assay'))
        return importlib.import_module('reference_feature_assay')

    def test_identity_directions_and_no_mutation(self):
        api=self.api();rgb=np.full((4,5,3),128,dtype=np.uint8);before=rgb.copy()
        self.assertEqual(len(api.OPERATIONS),11)
        self.assertTrue(np.array_equal(api.transform(rgb,'identity'),rgb))
        warm=api.transform(rgb,'warm');cool=api.transform(rgb,'cool')
        self.assertGreater(int(warm[0,0,0]),128);self.assertLess(int(warm[0,0,2]),128)
        self.assertLess(int(cool[0,0,0]),128);self.assertGreater(int(cool[0,0,2]),128)
        self.assertTrue(np.array_equal(rgb,before))
        gray=np.full((4,5,3),77,dtype=np.uint8)
        self.assertTrue(np.array_equal(api.transform(gray,'saturation_high'),gray))

    def test_split_tone_direction_and_validation(self):
        api=self.api();rgb=np.array([[[40,40,40],[210,210,210]]],dtype=np.uint8)
        out=api.transform(rgb,'cool_shadows')
        self.assertLess(out[0,0,0],out[0,0,2]);self.assertGreater(out[0,1,0],out[0,1,2])
        for invalid in [rgb.astype(float),rgb[:,:,:2],np.empty((0,1,3),dtype=np.uint8)]:
            with self.assertRaises(ValueError):api.transform(invalid,'identity')
        with self.assertRaises(ValueError):api.transform(rgb,'unknown')

    def test_moments_gram_and_pixel_permutation(self):
        api=self.api();x=torch.arange(24,dtype=torch.float64).reshape(1,3,2,4)/10
        mean,std,gram=api.appearance_statistics(x)
        self.assertTrue(torch.equal(mean,x.mean((-2,-1))))
        self.assertTrue(torch.allclose(std,x.flatten(2).std(-1,unbiased=False)))
        flat=x.flatten(2);full=flat@flat.transpose(1,2)/8;i,j=torch.triu_indices(3,3)
        self.assertTrue(torch.equal(gram,full[:,i,j]))
        for a,b in zip((mean,std,gram),api.appearance_statistics(x.flip(-1))):self.assertTrue(torch.allclose(a,b))
        with self.assertRaises(ValueError):api.appearance_statistics(torch.full((1,3,2,2),float('nan')))

    def test_group_folds_keep_all_operations_together(self):
        api=self.api();sources=[f'photo-{i}' for i in range(8)];folds=api.group_folds(sources)
        self.assertEqual(folds,api.group_folds(list(reversed(sources))));self.assertEqual(set(folds),set(sources))
        self.assertEqual(sorted(folds.values()),[0]*4+[1]*4)
        with self.assertRaises(ValueError):api.group_folds(['a','a','b','c'])

    def test_ridge_uses_train_statistics_and_is_finite(self):
        api=self.api();train=np.array([[0.,0.],[0.,1.],[2.,0.],[2.,1.]]);labels=np.array([0,0,1,1]);test=np.array([[0.,.5],[2.,.5]])
        pred=api.ridge_predict(train,labels,test,2)
        self.assertTrue(np.array_equal(pred,[0,1]))
        expanded=api.ridge_predict(train,labels,np.vstack([test,[10000.,-10000.]]),2)
        self.assertTrue(np.array_equal(pred,expanded[:2]))
        with self.assertRaises(ValueError):api.ridge_predict(train,labels,np.full((1,2),np.nan),2)


if __name__=='__main__':unittest.main()
