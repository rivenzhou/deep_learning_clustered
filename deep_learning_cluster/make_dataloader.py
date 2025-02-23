import functools
import itertools
from typing import Any, Union

import numpy as np
import torch

import torchtuples
from torchtuples.tupletree import tuplefy

def make_dataloader(
    data, batch_size, shuffle, num_workers=0, to_tensor=True, make_dataset=None, torch_ds_dl=False
):
    """Create a dataloder from tensor or np.arrays.
    Arguments:
        data {tuple, np.array, tensor} -- Data in dataloader e.g. (x, y)
        batch_size {int} -- Batch size used in dataloader
        shuffle {bool} -- If order should be suffled
    Keyword Arguments:
        num_workers {int} -- Number of workers in dataloader (default: {0})
        to_tensor {bool} -- Ensure that we use tensors (default: {True})
        make_dataset {callable} -- Function for making dataset. If 'None' we use
            DatasetTuple. (default {None}).
        torch_ds_dl {bool} -- If `True` we TensorDataset and DataLoader from torch. If
            `False` we use the (faster) versions from torchtuple (default {False}).
    Returns:
        DataLoaderBatch -- A dataloader object like the torch DataLoader
    """
    if to_tensor:
        data = tuplefy(data).to_tensor()
    if make_dataset is None:
        make_dataset = torchtuples.data.DatasetTuple
        if torch_ds_dl:
            make_dataset = torch.utils.data.TensorDataset
            # make_dataset = torchtuples.data.DatasetTupleSingle
    # dataset = DatasetTuple(data)
    dataset = make_dataset(*data)
    DataLoader = torchtuples.data.DataLoaderBatch
    if torch_ds_dl:
        DataLoader = torch.utils.data.DataLoader
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader