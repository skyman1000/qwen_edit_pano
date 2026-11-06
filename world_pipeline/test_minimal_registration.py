"""CPU regression tests for disjoint registration partitions and unchanged gates."""
import unittest
import numpy as np

from .minimal_world_sources import fit_registration_partition, unit


class RegistrationTests(unittest.TestCase):
    def make(self, counts):
        observations = [dict(yaw_index=index) for index in range(6)]
        groups = np.concatenate([np.full(count,index,dtype=int) for index,count in enumerate(counts)])
        a = unit(np.random.default_rng(87).normal(size=(len(groups),3)))
        basis = np.diag([1.,1.,-1.])
        return a,a@basis.T,groups,observations,basis

    def test_alternative_partition_preserves_independent_validation_and_handedness(self):
        a,b,groups,observations,basis = self.make([17,53,17,53,17,53])
        with self.assertRaisesRegex(ValueError,'Weak held-out'):
            fit_registration_partition(a,b,groups,observations,30,0)
        actual,report = fit_registration_partition(a,b,groups,observations,30,1)
        np.testing.assert_allclose(actual,basis,atol=1e-8)
        self.assertLess(np.linalg.det(actual),0)
        self.assertEqual((report['fit_inliers'],report['heldout_inliers'],report['heldout_observations']),(159,51,3))

    def test_alternative_partition_does_not_accept_insufficient_validation(self):
        a,b,groups,observations,_ = self.make([5,53,5,53,5,53])
        with self.assertRaisesRegex(ValueError,'Weak held-out'):
            fit_registration_partition(a,b,groups,observations,30,1)
        a,b,groups,observations,_ = self.make([51,53,0,53,0,53])
        with self.assertRaisesRegex(ValueError,'Weak held-out'):
            fit_registration_partition(a,b,groups,observations,30,1)

    def test_heldout_changes_do_not_fit_the_rotation(self):
        a,b,groups,observations,basis = self.make([35,40,35,40,35,40])
        first,_ = fit_registration_partition(a,b,groups,observations,30,0)
        changed=np.flatnonzero(groups==1)[:10]
        b[changed] = unit(np.random.default_rng(15).normal(size=(10,3)))
        second,_ = fit_registration_partition(a,b,groups,observations,30,0)
        np.testing.assert_allclose(first,second,atol=1e-10)


if __name__=='__main__':
    unittest.main()
