# -*- coding: utf-8 -*-

"""mtwi2018 预处理模块。"""

import math
import numpy as np

from skimage import io, transform


def read_img(file_name):
    """加载图片。

    Args:
        file_name: 文件路径。

    Returns:
        返回图片的ndarray数组。
    """
    return io.imread(file_name)

def save_img(file_name, arr):
    """保存图片。

    Args:
        file_name: 文件路径.
    """
    io.imsave(file_name, arr)


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
            content = ','.join(temp[8:])
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


def inverted_y(coordinate_y, height):
    """反转y坐标。

    Args:
        y: y坐标。
        height: 高度。

    Returns:
        新的y坐标
    """
    return abs(coordinate_y - height)

def crop_text_img(img, vertices):
    """裁剪图片。

    用顶点裁剪图片

    Args:
        vertices: 一个数组，存放了四边形的四个顶点。

    Returns:
        返回裁剪后的图片，类型为ndarray数组。
    """
    vertices = np.array(vertices).reshape(4, 2)
    vertices = sort_rectangle_vertices(vertices)

    top_left_point = vertices[0]
    bottom_right_point = vertices[2]
    bottom_left_point = vertices[3]

    radians = 0
    if(top_left_point - bottom_left_point != 0).all():
        radians = calculate_angles(top_left_point, bottom_left_point)

    print(math.degrees(radians), math.pi/18, math.pi * 17 / 18)
    if abs(radians) > math.radians(30) and abs(radians) < math.radians(60):
        rotated_img = transform.rotate(img, math.degrees(radians), True)
        print(np.abs(top_left_point - bottom_left_point) > 3)

        offset = [0, 0, 0]
        offset[0] = abs(rotated_img.shape[1] - img.shape[1])/2
        offset[1] = abs(rotated_img.shape[0] - img.shape[0])/2
        offset = np.array(offset).reshape(3, 1)

        matrix = rotation_matrix(
            math.sin(radians),
            math.cos(radians),
            img.shape[1]/2,
            img.shape[0]/2
        )

        top_left_point = np.append(top_left_point, [1]).reshape(3, 1)
        bottom_right_point = np.append(bottom_right_point, [1]).reshape(3, 1)

        top_left_point = np.round((matrix.dot(top_left_point) + offset)).reshape(3)
        bottom_right_point = np.round((matrix.dot(bottom_right_point) + offset)).reshape(3)

        img = rotated_img

    bounds = compute_bounds(top_left_point, bottom_right_point)
    print(img.shape)
    print(bounds[0], bounds[2])
    return img[bounds[1]:bounds[3], bounds[0]:bounds[2]]


def compute_bounds(top_left, bottom_right):
    """计算矩形边界。

    通过矩形左上角和右下角的点，计算包围盒。

    Args:
        top_left: 左上角。
        bottom_right: 右下角。

    Returns:
        返回包围盒
    """
    top_x = int(top_left[0] + 0.5)
    top_y = int(top_left[1] + 0.5)
    bottom_x = int(bottom_right[0] + 0.5)
    bottom_y = int(bottom_right[1] + 0.5)

    start_x = min(top_x, bottom_x)
    start_y = min(top_y, bottom_y)
    end_x = start_x + abs(top_x - bottom_x)
    end_y = start_y + abs(top_y - bottom_y)

    return np.array([start_x, start_y, end_x, end_y])
