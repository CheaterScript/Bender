# -*- coding: utf-8 -*-

"""基于H5PY的HDF5读写模块。"""

import h5py


def create(file):
    """创建HDF5文件。

    Args:
        file: 文件路径。
    """
    h5_file = h5py.File(file, mode='w')
    h5_file.close()


def create_group(file, name):
    """创建Group

    Args:
        file: 文件路径。
        name: Group名称。
    """
    with h5py.File(file, mode='a') as h5_file:
        h5_file.create_group(name)


def create_dataset(file, name, group=None, shape=None, dtype=None, data=None, **kwds):
    """创建Dataset。

    Args:
        file: 文件路径。
        name: Dataset名称。
        shape: 数据集的形状。
        dtype: 数据类型。
        kwds: 更多请参考h5py文档。
    """
    with h5py.File(file, mode='a') as h5_file:
        if group is not None:
            h5_file = h5_file[group]
        h5_file.create_dataset(name, shape, dtype, data, **kwds)


def append(file, data, path):
    """追加写入。

    Args:
        file: 文件路径。
        data: 数据。
        path: dataset路径。
    """
    with h5py.File(file, mode='a') as h5_file:
        dataset = h5_file[path]

        start = dataset.shape[0]
        data_shape = list(data.shape)
        dataset_shape = list(dataset.shape)

        dataset_shape[0] += data_shape[0]
        dataset.resize(dataset_shape)

        dataset[start: dataset_shape[0]] = data


def read(file, path, start=0, end=0):
    """读取数据。

    Args:
        file: 文件路径。
        path: dataset路径。
        start: 切片起点。
        end: 切片终点。

    Returns:
        返回数据。
    """
    with h5py.File(file, mode='a') as h5_file:
        if end > start:
            dataset = h5_file[path][start:end]
        else:
            dataset = h5_file[path][start:]
    return dataset


def len(file, path):
    """获取dataset第一维长度。

    Args:
        file: 文件路径。
        path: dataset路径。

    Returns:
        返回长度。
    """
    with h5py.File(file, mode='r') as h5_file:
        length = h5_file[path].len()
    return length