import unittest

import numpy as np
import torch
from selene_sdk.samplers.samples_batch import SamplesBatch


class TestSamplesBatch(unittest.TestCase):
    def test_single_input(self):
        samples_batch = SamplesBatch(np.ones((6, 200, 4)), target_batch=np.ones(20))

        inputs = samples_batch.inputs()
        self.assertIsInstance(inputs, np.ndarray)
        self.assertSequenceEqual(inputs.shape, (6, 200, 4))

        torch_inputs, _ = samples_batch.torch_inputs_and_targets(use_cuda=False)
        self.assertIsInstance(torch_inputs, torch.Tensor)
        self.assertSequenceEqual(torch_inputs.shape, (6, 4, 200))

    def test_multiple_inputs(self):
        samples_batch = SamplesBatch(
            np.ones((6, 200, 4)),
            other_input_batches={"something": np.ones(10)},
            target_batch=np.ones(20),
        )

        inputs = samples_batch.inputs()
        self.assertIsInstance(inputs, dict)
        self.assertEqual(len(inputs), 2)
        self.assertSequenceEqual(inputs["sequence_batch"].shape, (6, 200, 4))

        torch_inputs, _ = samples_batch.torch_inputs_and_targets(use_cuda=False)
        self.assertIsInstance(torch_inputs, dict)
        self.assertEqual(len(torch_inputs), 2)
        self.assertSequenceEqual(torch_inputs["sequence_batch"].shape, (6, 4, 200))

    def test_has_target(self):
        samples_batch = SamplesBatch(np.ones((6, 200, 4)), target_batch=np.ones(20))
        targets = samples_batch.targets()
        self.assertIsInstance(targets, np.ndarray)
        _, torch_targets = samples_batch.torch_inputs_and_targets(use_cuda=False)
        self.assertIsInstance(torch_targets, torch.Tensor)

    def test_no_target(self):
        samples_batch = SamplesBatch(np.ones((6, 200, 4)))
        self.assertIsNone(samples_batch.targets())
        _, torch_targets = samples_batch.torch_inputs_and_targets(use_cuda=False)
        self.assertIsNone(torch_targets)
