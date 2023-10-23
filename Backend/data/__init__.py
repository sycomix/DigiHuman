"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    dataset_filename = f"data.{dataset_name}_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    dataset = None
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError(
            f"In {dataset_filename}.py, there should be a subclass of BaseDataset with class name that matches {target_dataset_name} in lowercase."
        )

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    print("dataset: ",dataset)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" % (type(instance).__name__, len(instance)))
    return torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain,
    )
