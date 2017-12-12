import math
from collections import Counter


def init_means(num_means, value_counter):
    # Divide values equally
    sorted_lengths = list()
    for length, count in value_counter.iteritems():
        sorted_lengths.extend([length] * count)
    sorted_lengths.sort()
    size = int(math.ceil(1. * len(sorted_lengths) / num_means))
    splits = list()
    for k in range(num_means):
        splits.append(sorted_lengths[k * size: (k + 1) * size][-1])

    # Make means unique
    k = len(splits) - 1
    while k > 0:
        while splits[k - 1] >= splits[k] or splits[k - 1] not in value_counter:
            splits[k - 1] -= 1
            if splits[k - 1] == 0:
                break
        k -= 1
    while splits[0] not in value_counter:
        splits[0] += 1
    k = 1
    while k < len(splits):
        while splits[k] <= splits[k - 1] or splits[k] not in value_counter:
            splits[k] += 1
        k += 1
    return splits


def recenter(splits, unique_values, value_counter, split_counter, split_idx2value_idx):
    for split_idx in range(len(splits) - 1):
        split = splits[split_idx]
        right_split = splits[split_idx + 1]
        value_idx = split_idx2value_idx[split]

        if value_idx > 0 and unique_values[value_idx - 1] not in split_counter:
            # Try to move split to the left
            # new_split --- split --- right_split
            new_split = unique_values[value_idx - 1]
            old_distance = abs(split_counter[right_split] - split_counter[split])
            new_distance = abs(split_counter[right_split] - split_counter[split] + 2 * value_counter[split])
            if old_distance > new_distance:
                splits[split_idx] = new_split
                split_idx2value_idx[new_split] = value_idx - 1
                del split_idx2value_idx[split]
                split_counter[new_split] = split_counter[split] - value_counter[split]
                split_counter[right_split] += value_counter[split]
                del split_counter[split]
        elif value_idx < len(unique_values) - 2 and unique_values[value_idx + 1] not in split_counter:
            # Try to move split to the right
            # split --- new_split --- right_split
            new_split = unique_values[value_idx + 1]
            old_distance = abs(split_counter[right_split] - split_counter[split])
            new_distance = abs(split_counter[right_split] - split_counter[split] - 2 * value_counter[split])
            if old_distance > new_distance:
                splits[split_idx] = new_split
                split_idx2value_idx[new_split] = value_idx + 1
                del split_idx2value_idx[split]
                split_counter[new_split] = split_counter[split] + value_counter[split]
                split_counter[right_split] -= value_counter[split]
                del split_counter[split]


def kmean(num_means, vector):
    value_counter = Counter()
    for value in vector:
        value_counter[value] += 1
    if len(value_counter) < num_means:
        raise ValueError('Trying to sort %d data points into %d buckets' % (len(value_counter), num_means))
    sorted_values = sorted(value_counter.keys())

    # Initialize means
    splits = init_means(num_means, value_counter)

    # Indexing
    split_idx2value_idx = dict()
    split_counter = Counter()
    split_idx = 0
    split = splits[split_idx]
    for value_idx, value in enumerate(sorted_values):
        split_counter[split] += value_counter[value]
        if value == splits[split_idx]:
            split_idx2value_idx[split] = value_idx
            split_idx += 1
            if split_idx < len(splits):
                split = splits[split_idx]

    # Re-centering
    old_splits = None
    while old_splits != splits:
        old_splits = list(splits)
        recenter(splits, sorted_values, value_counter, split_counter, split_idx2value_idx)

    # Re-indexing
    value2split = dict()
    split_idx = 0
    for value_idx, value in enumerate(sorted_values):
        value2split[value] = split_idx
        if value == splits[split_idx]:
            split_idx += 1
    return value2split, splits
