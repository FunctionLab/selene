import numpy as np
import torch


class SamplesBatch:
    """
    This class represents NN inputs and targets. Values are stored as numpy.ndarrays
    and there is a method to convert them to torch.Tensors.

    Inputs are stored in a dict, which can be used if you are providing more than just a
    `sequence_batch` to the NN.

    NOTE: If you store just a sequence as an input to the model, then `inputs()` and
    `torch_inputs_and_targets()` will return only the batch of sequences rather than
    a dict.

    """

    _SEQUENCE_LABEL = "sequence_batch"

    def __init__(
        self,
        sequence_batch: np.ndarray,
        other_input_batches=dict(),
        target_batch: np.ndarray = None,
    ) -> None:
        self._input_batches = other_input_batches.copy()
        self._input_batches[self._SEQUENCE_LABEL] = sequence_batch
        self._target_batch = target_batch

    def sequence_batch(self) -> torch.Tensor:
        """Returns the sequence batch with a shape of
        [batch_size, sequence_length, alphabet_size].
        """
        return self._input_batches[self._SEQUENCE_LABEL]

    def inputs(self):
        """Based on the size of inputs dictionary, returns either just the
        sequence or the whole dictionary.

        Returns
        -------
        numpy.ndarray or dict[str, numpy.ndarray]
            numpy.ndarray is returned when inputs contain just the sequence batch.
            dict[str, numpy.ndarray] is returned when there are multiple inputs.

            NOTE: Sequence batch has a shape of
                [batch_size, sequence_length, alphabet_size].
        """
        if len(self._input_batches) == 1:
            return self.sequence_batch()

        return self._input_batches

    def targets(self):
        """Returns target batch if it is present.

        Returns
        -------
        numpy.ndarray

        """
        return self._target_batch

    def torch_inputs_and_targets(self, use_cuda: bool):
        """
        Returns inputs and targets in torch.Tensor format.

        Based on the size of inputs dictionary, returns either just the
        sequence or the whole dictionary.

        Returns
        -------
        inputs, targets :\
                tuple(numpy.ndarray or dict[str, numpy.ndarray], numpy.ndarray)
            For `inputs`:
                numpy.ndarray is returned when inputs contain just the sequence batch.
                dict[str, numpy.ndarray] is returned when there are multiple inputs.

                NOTE: Returned sequence batch has a shape of
                    [batch_size, alphabet_size, sequence_length].

        """
        all_inputs = dict()
        for key, value in self._input_batches.items():
            all_inputs[key] = torch.Tensor(value)

            if use_cuda:
                all_inputs[key] = all_inputs[key].cuda()

        # Transpose the sequences to satisfy NN convolution input format (which is
        # [batch_size, channels_size, sequence_length]).
        all_inputs[self._SEQUENCE_LABEL] = all_inputs[self._SEQUENCE_LABEL].transpose(
            1, 2
        )

        inputs = all_inputs if len(all_inputs) > 1 else all_inputs[self._SEQUENCE_LABEL]

        targets = None
        if self._target_batch is not None:
            targets = torch.Tensor(self._target_batch)

            if use_cuda:
                targets = targets.cuda()

        return inputs, targets
