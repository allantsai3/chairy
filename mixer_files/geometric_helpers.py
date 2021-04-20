import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
Helper function to transform 3D affine point clouds
'''


def plot_boundaries(ax, x, y, z, color):
    ax.plot([x[0], x[1]], [y[0], y[1]],
            [z[0], z[1]], c=color)
    ax.plot([x[0], x[2]], [y[0], y[2]],
            [z[0], z[2]], c=color)
    ax.plot([x[1], x[3]], [y[1], y[3]],
            [z[1], z[3]], c=color)
    ax.plot([x[2], x[3]], [y[2], y[3]],
            [z[2], z[3]], c=color)
    ax.plot([x[4], x[5]], [y[4], y[5]],
            [z[4], z[5]], c=color)
    ax.plot([x[4], x[6]], [y[4], y[6]],
            [z[4], z[6]], c=color)
    ax.plot([x[5], x[7]], [y[5], y[7]],
            [z[5], z[7]], c=color)
    ax.plot([x[6], x[7]], [y[6], y[7]],
            [z[6], z[7]], c=color)
    ax.plot([x[0], x[4]], [y[0], y[4]],
            [z[0], z[4]], c=color)
    ax.plot([x[1], x[5]], [y[1], y[5]],
            [z[1], z[5]], c=color)
    ax.plot([x[2], x[6]], [y[2], y[6]],
            [z[2], z[6]], c=color)
    ax.plot([x[3], x[7]], [y[3], y[7]],
            [z[3], z[7]], c=color)


def plot_bounding_box(bounding_box1, *argv):
    fig = plt.figure()
    cmap = plt.get_cmap('jet_r')
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    i1 = np.array(bounding_box1)
    x1, y1, z1 = i1.T

    plot_boundaries(ax, x1, y1, z1, 'blue')

    for i in range(len(argv)):
        i2 = np.array(argv[i])
        x2, y2, z2 = i2.T

        plot_boundaries(ax, x2, y2, z2, cmap(float(i) / len(argv)))

    plt.show()


def bounding_box_transform(set1, set2):
    src = np.array([set1], copy=True).astype(np.float32)
    dst = np.array([set2], copy=True).astype(np.float32)

    t = cv2.estimateAffine3D(src, dst, confidence=0.99)

    return t[1]


def agg_boxes(box_arr):
    # print(box_arr[0])
    # plot_bounding_box(box_arr[0], box_arr[1])
    # Instantiate the bounding box with the first point in the bounding box array
    pos_x = box_arr[0][0][0]
    neg_x = box_arr[0][0][0]
    pos_y = box_arr[0][0][1]
    neg_y = box_arr[0][0][1]
    pos_z = box_arr[0][0][2]
    neg_z = box_arr[0][0][2]

    for box in box_arr:
        for point in box:
            if point[0] < neg_x:
                neg_x = point[0]
            elif point[0] > pos_x:
                pos_x = point[0]

            if point[1] < neg_y:
                neg_y = point[1]
            elif point[1] > pos_y:
                pos_y = point[1]

            if point[2] < neg_z:
                neg_z = point[2]
            elif point[2] > pos_z:
                pos_z = point[2]

    new_bounding_box = [[pos_x, pos_y, pos_z], [pos_x, neg_y, pos_z], [neg_x, pos_y, pos_z], [neg_x, neg_y, pos_z],
                        [pos_x, pos_y, neg_z], [pos_x, neg_y, neg_z], [neg_x, pos_y, neg_z], [neg_x, neg_y, neg_z]]

    # plot_bounding_box(box_arr[0], box_arr[1], new_bounding_box)

    return new_bounding_box


def shift_vertices(part, part_name, magnitude):
    new_vert = []
    if part_name == 'chair_back':
        for vert in part.get_vertices():
            new_vert.append((vert[0], vert[1] + magnitude, vert[2]))

    return new_vert
