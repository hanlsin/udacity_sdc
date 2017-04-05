import math
from pprint import pprint

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    batch_cnt = int(len(features) / batch_size)
    exist_rest = False
    if len(features) % batch_size != 0:
        exist_rest = True

    batch_arr = []
    for i in range(int(batch_cnt)):
        arr = []
        start_idx = i * batch_size
        arr.append(features[start_idx:start_idx + batch_size])
        arr.append(labels[start_idx:start_idx + batch_size])
        batch_arr.append(arr)

    if exist_rest:
        arr = []
        start_idx = batch_cnt * batch_size
        arr.append(features[start_idx:len(features)])
        arr.append(labels[start_idx:len(features)])
        batch_arr.append(arr)
    return batch_arr
    pass

def batches_solution(batch_size, features, labels):
    """
        Create batches of features and labels
        :param batch_size: The batch size
        :param features: List of features
        :param labels: List of labels
        :return: Batches of (Features, Labels)
        """
    assert len(features) == len(labels)
    # TODO: Implement batching
    output_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)

    return output_batches

# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]

"""
The example_batches variable would be the following:
[
    # 2 batches:
    #   First is a batch of size 3.
    #   Second is a batch of size 1
    [
        # First Batch is size 3
        [
            # 3 samples of features.
            # There are 4 features per sample.
            ['F11', 'F12', 'F13', 'F14'],
            ['F21', 'F22', 'F23', 'F24'],
            ['F31', 'F32', 'F33', 'F34']
        ], [
            # 3 samples of labels.
            # There are 2 labels per sample.
            ['L11', 'L12'],
            ['L21', 'L22'],
            ['L31', 'L32']
        ]
    ], [
        # Second Batch is size 1.
        # Since batch size is 3, there is only one sample left from the 4 samples.
        [
            # 1 sample of features.
            ['F41', 'F42', 'F43', 'F44']
        ], [
            # 1 sample of labels.
            ['L41', 'L42']
        ]
    ]
]
"""
example_batches = batches(3, example_features, example_labels)
pprint(example_batches)
