import unittest
import sys
import numpy as np
import torch
import torch.nn as nn

from .MeshNet import MeshNet
from .layers import SpatialDescriptor, StructuralDescriptor, MeshConvolution, FaceKernelCorrelation

class TestModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfgs = []
        for aggr_method in ['Concat', 'Average', 'Max']:
            for num_kernel in [64, 42]:
                for max_faces in [1024, 123]:
                    for batch_size in [5, 1]:
                        cfg = {
                            'MeshNet': {
                                'structural_descriptor': {
                                    'num_kernel': num_kernel,
                                    'sigma': 0.2,
                                },
                                'mesh_convolution': {
                                    'aggregation_method': aggr_method,
                                }
                            },
                            'dataset': {
                                'max_faces': max_faces,
                            },
                            'batch_size': batch_size,
                        }
                        cls.cfgs.append(cfg)

    def setUp(self):
        torch.manual_seed(123)
        self.models = []
        for cfg in TestModules.cfgs:
            model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
            self.models.append(model)
        torch.manual_seed(123)

    def tearDown(self):
        pass

    def test_shape_model(self):
        for cfg, model in zip(TestModules.cfgs, self.models):
            n = cfg['batch_size']
            m = cfg['dataset']['max_faces']
            centers = torch.randn(n, 3, m)
            corners = torch.randn(n, 9, m)
            normals = torch.randn(n, 3, m)
            neighbor_index = torch.randint(0, n, (n, m, 3), dtype=torch.long)
            module = model
            outputs, feas = module(centers, corners, normals, neighbor_index)
            self.assertEqual(outputs.shape, (n, 40))
            self.assertEqual(feas.shape, (n, 256))

    def test_shape_spd_integrated(self):
        for cfg, model in zip(TestModules.cfgs, self.models):
            n = cfg['batch_size']
            m = cfg['dataset']['max_faces']

            centers = torch.randn(n, 3, m)

            module = model.spatial_descriptor
            outputs = module(centers)
            self.assertEqual(outputs.shape, (n, 64, m))

    def test_shape_std_integrated(self):
        for cfg, model in zip(TestModules.cfgs, self.models):
            n = cfg['batch_size']
            m = cfg['dataset']['max_faces']

            corners = torch.randn(n, 9, m)
            normals = torch.randn(n, 3, m)
            neighbor_index = torch.randint(0, n, (n, m, 3), dtype=torch.long)

            module = model.structural_descriptor
            outputs = module(corners, normals, neighbor_index)
            self.assertEqual(outputs.shape, (n, 131, m))

    def test_shape_frc_integrated(self):
        for cfg, model in zip(TestModules.cfgs, self.models):
            n = cfg['batch_size']
            m = cfg['dataset']['max_faces']

            corners = torch.randn(n, 9, m)

            module = model.structural_descriptor.FRC
            outputs = module(corners)
            self.assertEqual(outputs.shape, (n, 64, m))

    def test_shape_fkc(self):
        for n in [5, 1]:
            m = 1024
            sigma = 0.2222
            normals = torch.randn(n, 3, m)
            neighbor_index = torch.randint(0, n, (n, m, 3), dtype=torch.long)
            for num_kernel in [12, 23, 34, 45]:
                module = FaceKernelCorrelation(num_kernel, sigma)
                outputs = module(normals, neighbor_index)
                self.assertEqual(outputs.shape, (n, num_kernel, m))

    def test_shape_fkc_integrated(self):
        for cfg, model in zip(TestModules.cfgs, self.models):
            n = cfg['batch_size']
            m = cfg['dataset']['max_faces']
            num_kernel = cfg['MeshNet']['structural_descriptor']['num_kernel']

            normals = torch.randn(n, 3, m)
            neighbor_index = torch.randint(0, n, (n, m, 3), dtype=torch.long)

            module = model.structural_descriptor.FKC
            outputs = module(normals, neighbor_index)
            self.assertEqual(outputs.shape, (n, num_kernel, m))

    def test_shape_mc(self):
        for aggr_method in ['Max', 'Average', 'Concat']:
            cfg_mc = {'aggregation_method': aggr_method}
            n = 5
            m = 1024
            sp_in, st_in, sp_out, st_out = 123, 234, 345, 456
            spd_output = torch.randn(n, sp_in, m)
            std_output = torch.randn(n, st_in, m)
            neighbor_index = torch.randint(0, n, (n, m, 3), dtype=torch.long)

            module = MeshConvolution(cfg_mc, sp_in, st_in, sp_out, st_out)
            sp_fea, st_fea = module(spd_output, std_output, neighbor_index)
            self.assertEqual(sp_fea.shape, (n, sp_out, m))
            self.assertEqual(st_fea.shape, (n, st_out, m))

    def test_shape_mc1_integrated(self):
        for cfg, model in zip(TestModules.cfgs, self.models):
            n = cfg['batch_size']
            m = cfg['dataset']['max_faces']

            module = model.mesh_conv1

            self.assertEqual(module.spatial_in_channel,    64)
            self.assertEqual(module.structural_in_channel, 131)

            spd_output = torch.randn(n, module.spatial_in_channel,    m)
            std_output = torch.randn(n, module.structural_in_channel, m)
            neighbor_index = torch.randint(0, n, (n, m, 3), dtype=torch.long)

            self.assertEqual(module.spatial_in_channel,    spd_output.shape[1])
            self.assertEqual(module.structural_in_channel, std_output.shape[1])

            sp_fea, st_fea = module(spd_output, std_output, neighbor_index)

            self.assertEqual(sp_fea.shape, (n, module.spatial_out_channel,    m))
            self.assertEqual(st_fea.shape, (n, module.structural_out_channel, m))

    def test_shape_mc2_integrated(self):
        for cfg, model in zip(TestModules.cfgs, self.models):
            n = cfg['batch_size']
            m = cfg['dataset']['max_faces']

            module = model.mesh_conv2

            spd_output = torch.randn(n, module.spatial_in_channel,    m)
            std_output = torch.randn(n, module.structural_in_channel, m)
            neighbor_index = torch.randint(0, n, (n, m, 3), dtype=torch.long)

            self.assertEqual(module.spatial_out_channel,    512)
            self.assertEqual(module.structural_out_channel, 512)

            sp_fea, st_fea = module(spd_output, std_output, neighbor_index)

            self.assertEqual(sp_fea.shape, (n, module.spatial_out_channel,    m))
            self.assertEqual(st_fea.shape, (n, module.structural_out_channel, m))

if __name__ == "__main__":
	unittest.main()
