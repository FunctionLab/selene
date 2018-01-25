import numpy as np
import torch
from torch.autograd import Variable

def evaluate(model,
             batch_size,
             input_sequence_encodings,
             features_list,
             output_file):
    sequence_encodings = None
    if isinstance(input_sequence_encodings, str):
        from proteus import _sequence_to_encoding
        sequence_encodings = []
        with open(input_sequence_encodings, 'r') as file_handle:
            lines = file_handle.readlines()
            for line in lines:
                sequence_encoding = _sequence_to_encoding(
                    line.strip(), {'A': 0, 'C': 1, 'G': 2, 'T': 3})
                sequence_encodings.append(sequence_encoding)
        sequence_encoding = np.array(sequence_encoding)
    else:
        sequence_encodings = input_sequence_encodings

    make_predictions = open(output_file, 'w+')

    columns = ["line_num"]
    columns += features_list

    columns_str = ';'.join(columns)
    make_predictions.write("{0}\n".format(columns_str))

    n_batches = sequence_encodings.shape[0] / batch_size

    for batch_number in range(int(n_batches)):
        start = batch_number * batch_size
        end = (batch_number + 1) * batch_size
        sequences_batch = sequence_encodings[start:end, :, :]

        inputs = torch.Tensor(sequences_batch)
        inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)

        output = model(inputs.transpose(1, 2))
        predictions = output.data.cpu().numpy()

        for index, sample_predict in enumerate(predictions):
            row = [str(start + index)]
            sample_predict = sample_predict.tolist()
            sample_predict = ["{:.2e}".format(x) for x in sample_predict]
            row += sample_predict

            row_str = ','.join(row)
            make_predictions.write("{0}\n".format(row_str))

    make_predictions.close()

if __name__ == "__main__":
    resnet = "/tigress/kc31/outputs/2018-01-17-16-05-41/best_model.pth.tar"
    origin = "/tigress/kc31/outputs/2018-01-17-10-58-17/best_model.pth.tar"


