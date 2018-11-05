# -*- coding: utf-8 -*-

"""mtwi2018 预处理模块。"""

import math
import numpy as np

from skimage import io


def read_img(file_name):
    """加载图片。

    Args:
        file_name: 文件路径。

    Returns:
        返回图片的ndarray数组。
    """
    return io.imread(file_name)


def read_txt(file_name):
    """加载文本。

    Args:
        file_name: 文件路径。

    Returns:
        返回解析文件后得到的list。
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        data = file.readlines()

        rows = []
        for line in data:
            line = line.strip('\n')
            temp = line.split(',')
            box = [float(x) for x in temp[0:8]]
            content = ''.join(temp[8:])
            rows.append([box, content])

        return rows


def rotation_matrix(sin, cos, tx_value, ty_value):
    """生成旋转矩阵。

    Args:
        sin: 正弦。
        cos: 余弦。
        tx_value: 偏移点横坐标。
        ty_value: 偏移点纵坐标。

    Returns:
        返回一个ndarray数组。
    """
    return np.array([
        [cos, -sin, (1 - cos) * tx_value + ty_value * sin],
        [sin, cos, (1 - cos)* ty_value - tx_value * sin],
        [0, 0, 1]
    ])


def sort_rectangle_vertices(vertices):
    """排序矩形的四个顶点。

    从左上角按顺时针顺序排序。

    Args:
        vertices: 一个形状为(4, 2)的ndarray数组，存放矩形的四个顶点。

    Returns:
        返回排序后的顶点数组
    """
    vertices_sum = np.sum(vertices, axis=0)
    center = vertices_sum / 4

    top = []
    bottom = []
    for i in range(vertices.shape[0]):
        item = vertices[i, :]
        if item[1] >= center[1]:
            top.append(item)
        else:
            bottom.append(item)

    top.sort(key=lambda item: item[0])
    bottom.sort(key=lambda item: item[0], reverse=True)

    return np.array(top + bottom)


def calculate_angles(start, end):
    """求两个点连线和水平面的锐角夹角。

    Args:
        start: 起点。
        end: 终点。

    Returns:
        返回夹角弧度。
    """
    vector = end - start
    return math.atan(vector[1] / vector[0])
    