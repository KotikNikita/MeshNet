import unittest
import sys
import numpy as np
import torch

from .ModelNet40 import ModelNet40

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfgs = []
        for augment_data in [False, True]:
            for part in ['train', 'test']:
                cfg = {
                    'dataset': {
                        'data_root': '../ModelNet40_MeshNet/',
                        'augment_data': augment_data,
                        'max_faces': 1024,
                    }
                }
            cls.cfgs.append((cfg, part))

    def setUp(self):
        self.datasets = []
        for cfg, part in TestDataset.cfgs:
            np.random.seed(123)
            torch.manual_seed(123)
            dataset = ModelNet40(cfg=cfg['dataset'], part=part)
            self.datasets.append(dataset)
        np.random.seed(123)
        torch.manual_seed(123)

    def tearDown(self):
        pass

    def test_shape(self):
        for (cfg, part), dataset in zip(TestDataset.cfgs, self.datasets):
            m = cfg['dataset']['max_faces']
            n_chosen = 50
            idx_chosen = np.random.randint(0, len(dataset), (n_chosen))
            for idx in idx_chosen:
                centers, corners, normals, neighbor_index, target = dataset[idx]
                self.assertEqual(centers.shape, (3, m))
                self.assertEqual(corners.shape, (9, m))
                self.assertEqual(normals.shape, (3, m))
                self.assertEqual(neighbor_index.shape, (m, 3))
                self.assertEqual(target.shape, torch.Size([]))
                self.assertTrue((neighbor_index >= 0).all())
                self.assertTrue((neighbor_index < m).all())

if __name__ == "__main__":
	unittest.main()
