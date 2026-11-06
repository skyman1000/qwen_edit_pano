import unittest

from .region_projection import closest_region_center


class RegionProjectionTests(unittest.TestCase):
    def test_actual_v2_numeric_shape_has_nearest_candidate_beyond_old_radius(self):
        r=closest_region_center([1.5,.05,3.2],[.08,.12,.08],'right',90,[1024,1024])
        self.assertAlmostEqual(r['center'][0],2.3900005)
        self.assertAlmostEqual(r['center'][2],2.3099995)
        self.assertGreater(r['translation_m'][0],.75)
        self.assertFalse(r['input_frustum_intersects'])
        self.assertFalse(r['semantic_validity_verified'])

    def test_already_feasible_box_is_unchanged(self):
        for center,region in [([4.,0.,2.],'right'),([-4.,0.,2.],'left'),
                              ([0.,0.,-3.],'behind_camera'),([0.,4.,2.],'above'),([0.,-4.,2.],'below')]:
            r=closest_region_center(center,[.5,.5,.5],region,90,[100,100])
            self.assertEqual(r['center'],center)
            self.assertEqual(r['translation_norm_m'],0.)

    def test_projection_certifies_side_and_vertical_cones_with_aspect(self):
        for region in ('left','right','behind_camera','above','below'):
            r=closest_region_center([0.,0.,3.],[1.,1.,1.],region,90,[200,100])
            self.assertFalse(r['input_frustum_intersects'])
            self.assertGreater(r['region_margin_m'],0.)


if __name__=='__main__':unittest.main()
