import math
from copy import deepcopy

import cv2
import numpy as np
import rospy
from atom_core.dataset_io import genCollectionPrefix, getPointsInSensorAsNPArray, getPointsInSensorAsNPArrayNonCached
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Pose, Vector3, Quaternion

# own packages
from atom_core.config_io import uriReader


def draw_concentric_circles(plt, radii, color='tab:gray'):
    for r in radii:  # meters
        x = []
        y = []
        for theta in np.linspace(0, 2 * math.pi, num=150):
            x.append(r * math.cos(theta))
            y.append(r * math.sin(theta))
        plt.plot(x, y, '-', color='tab:gray')

        x = r * math.cos(math.pi / 2)
        y = r * math.sin(math.pi / 2)
        plt.text(x + .5, y, str(r) + ' (m)', color='tab:gray')


def draw_2d_axes(plt):
    plt.ylabel('y')
    plt.xlabel('x')
    ax = plt.axes()
    ax.arrow(0, 0, 10, 0, head_width=.5, head_length=1, fc='r', ec='r')
    ax.arrow(0, 0, 0, 10, head_width=.5, head_length=1, fc='g', ec='g')

    plt.text(10, 0.5, 'X', color='red')
    plt.text(-1.0, 10, 'Y', color='green')


def colormapToRVizColor(color):
    """ Converts a Matbplotlib colormap into an rviz display color format."""
    return str(int(color[0] * 255)) + '; ' + str(int(color[1] * 255)) + '; ' + str(
        int(color[2] * 255))


def drawSquare2D(image, x, y, size, color=(0, 0, 255), thickness=1):
    """
    Draws a square on the image
    :param image:
    :param x:
    :param y:
    :param color:
    :param thickness:
    """

    h, w, _ = image.shape
    if x - size < 0 or x + size >= w or y - size < 0 or y + size >= h:
        # print("Cannot draw square")
        return None

    # tl, tr, bl, br -> top left, top right, bottom left, bottom right
    tl = (int(x - size), int(y - size))
    tr = (int(x + size), int(y - size))
    br = (int(x + size), int(y + size))
    bl = (int(x - size), int(y + size))

    # cv2.line(image, (x,y), (x,y), color, 5)
    cv2.line(image, tl, tr, color, thickness)
    cv2.line(image, tr, br, color, thickness)
    cv2.line(image, br, bl, color, thickness)
    cv2.line(image, bl, tl, color, thickness)


def drawCross2D(image, x, y, size, color=(0, 0, 255), thickness=1):
    """
    Draws a square on the image
    :param image:
    :param x:
    :param y:
    :param color:
    :param thickness:
    """

    h, w, _ = image.shape
    if x - size < 0 or x + size > w or y - size < 0 or y + size > h:
        # print("Cannot draw square")
        return None

    # tl, tr, bl, br -> top left, top right, bottom left, bottom right
    left = (int(x - size), int(y))
    right = (int(x + size), int(y))
    top = (int(x), int(y - size))
    bottom = (int(x), int(y + size))

    cv2.line(image, left, right, color, thickness)
    cv2.line(image, top, bottom, color, thickness)


def drawLabelsOnImage(labels, image, color_idxs=(0, 200, 255), color_idxs_limits=(255, 0, 200)):
    _, width, _ = image.shape

    for idx in labels['idxs']:
        # convert from linear idx to x_pix and y_pix indices.
        y = int(idx / width)
        x = int(idx - y * width)
        cv2.line(image, (x, y), (x, y), color_idxs, 3)

    for idx in labels['idxs_limit_points']:
        # convert from linear idx to x_pix and y_pix indices.
        y = int(idx / width)
        x = int(idx - y * width)
        cv2.line(image, (x, y), (x, y), color_idxs_limits, 3)

    return image


def createPatternMarkers(frame_id, ns, collection_key, now, dataset, graphics):
    markers = MarkerArray()

    # Draw pattern frame lines_sampled (top, left, right, bottom)
    marker = Marker(header=Header(frame_id=frame_id, stamp=now),
                    ns=ns + '-frame_sampled', id=0, frame_locked=True,
                    type=Marker.CUBE_LIST, action=Marker.ADD, lifetime=rospy.Duration(0),
                    pose=Pose(position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
                    scale=Vector3(x=0.01, y=0.01, z=0.01),
                    color=ColorRGBA(r=graphics['collections'][collection_key]['color'][0],
                                    g=graphics['collections'][collection_key]['color'][1],
                                    b=graphics['collections'][collection_key]['color'][2], a=1.0))

    pts = []
    pts.extend(dataset['patterns']['frame']['lines_sampled']['left'])
    pts.extend(dataset['patterns']['frame']['lines_sampled']['right'])
    pts.extend(dataset['patterns']['frame']['lines_sampled']['top'])
    pts.extend(dataset['patterns']['frame']['lines_sampled']['bottom'])
    for pt in pts:
        marker.points.append(Point(x=pt['x'], y=pt['y'], z=0))

    markers.markers.append(marker)

    # Draw corners
    marker = Marker(header=Header(frame_id=frame_id, stamp=now),
                    ns=ns + '-corners', id=0, frame_locked=True,
                    type=Marker.CUBE_LIST, action=Marker.ADD, lifetime=rospy.Duration(0),
                    pose=Pose(position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
                    scale=Vector3(x=0.02, y=0.02, z=0.02),
                    color=ColorRGBA(r=graphics['collections'][collection_key]['color'][0],
                                    g=graphics['collections'][collection_key]['color'][1],
                                    b=graphics['collections'][collection_key]['color'][2], a=1.0))

    for idx_corner, pt in enumerate(dataset['patterns']['corners']):
        marker.points.append(Point(x=pt['x'], y=pt['y'], z=0))
        marker.colors.append(ColorRGBA(r=graphics['pattern']['colormap'][idx_corner, 0],
                                       g=graphics['pattern']['colormap'][idx_corner, 1],
                                       b=graphics['pattern']['colormap'][idx_corner, 2], a=1))

    markers.markers.append(marker)

    # Draw transitions
    # TODO we don't use this anymore, should we draw it? Perhaps it will be used for 2D Lidar ...
    # marker = Marker(header=Header(frame_id=frame_id, stamp=now),
    #                 ns=ns + '-transitions', id=0, frame_locked=True,
    #                 type=Marker.POINTS, action=Marker.ADD, lifetime=rospy.Duration(0),
    #                 pose=Pose(position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
    #                 scale=Vector3(x=0.015, y=0.015, z=0),
    #                 color=ColorRGBA(r=graphics['collections'][collection_key]['color'][0],
    #                                 g=graphics['collections'][collection_key]['color'][1],
    #                                 b=graphics['collections'][collection_key]['color'][2], a=1.0))
    #
    # pts = dataset['patterns']['transitions']['vertical']
    # pts.extend(dataset['patterns']['transitions']['horizontal'])
    # for pt in pts:
    #     marker.points.append(Point(x=pt['x'], y=pt['y'], z=0))
    #
    # markers.markers.append(marker)

    # Draw the mesh, if one is provided
    if not dataset['calibration_config']['calibration_pattern']['mesh_file'] == "":
        # rgba = graphics['collections'][collection_key]['color']
        # color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=1))

        # print('Got the mesh it is: ' + dataset['calibration_config']['calibration_pattern']['mesh_file'])
        m = Marker(header=Header(frame_id=frame_id, stamp=now),
                   ns=str(collection_key) + '-mesh', id=0, frame_locked=True,
                   type=Marker.MESH_RESOURCE, action=Marker.ADD, lifetime=rospy.Duration(0),
                   pose=Pose(position=Point(x=0, y=0, z=0),
                             orientation=Quaternion(x=0, y=0, z=0, w=1)),
                   scale=Vector3(x=1.0, y=1.0, z=1.0),
                   color=ColorRGBA(r=1, g=1, b=1, a=1))

        mesh_file, _, _ = uriReader(dataset['calibration_config']['calibration_pattern']['mesh_file'])
        m.mesh_resource = 'file://' + mesh_file  # mesh_resource needs uri format
        m.mesh_use_embedded_materials = True
        markers.markers.append(m)

    return markers  # return markers


def getRvizMarkersFrom3DLabels(dataset, collection_key, sensor_key, stamp, color, markers=None, use_cache=False):

    if dataset['sensors'][sensor_key]['modality'] not in ['lidar3d', 'depth']:
        raise ValueError('Sensor ' + sensor_key + ' does not have an adequate modality. ' +
                         'Can only generate rviz markers for modalities lidar3d or deph.')

    if markers is None:
        markers = MarkerArray()

    # Add labelled points to the marker
    collection = dataset['collections'][collection_key]
    frame_id = genCollectionPrefix(collection_key, collection['data'][sensor_key]['header']['frame_id'])
    marker = Marker(
        header=Header(frame_id=frame_id, stamp=stamp),
        ns=str(collection_key) + '-' + str(sensor_key), id=0, frame_locked=True,
        type=Marker.SPHERE_LIST, action=Marker.ADD, lifetime=rospy.Duration(0),
        pose=Pose(position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
        scale=Vector3(x=0.03, y=0.03, z=0.03),
        color=ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.5)
    )

    if use_cache:
        points = getPointsInSensorAsNPArray(collection_key, sensor_key, 'idxs', dataset)
    else:
        points = getPointsInSensorAsNPArrayNonCached(collection_key, sensor_key, 'idxs', dataset)

    for idx in range(0, points.shape[1]):
        marker.points.append(
            Point(x=points[0, idx], y=points[1, idx], z=points[2, idx]))

    markers.markers.append(deepcopy(marker))

    # Add limit points to the marker, this time with larger spheres
    marker = Marker(
        header=Header(frame_id=frame_id, stamp=stamp),
        ns=str(collection_key) + '-' + str(sensor_key) + '-limit_points', id=0, frame_locked=True,
        type=Marker.SPHERE_LIST, action=Marker.ADD, lifetime=rospy.Duration(0),
        pose=Pose(position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
        scale=Vector3(x=0.07, y=0.07, z=0.07),
        color=ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.5))

    if use_cache:
        points = getPointsInSensorAsNPArray(collection_key, sensor_key, 'idxs_limit_points', dataset)
    else:
        points = getPointsInSensorAsNPArrayNonCached(collection_key, sensor_key, 'idxs_limit_points', dataset)

    for idx in range(0, points.shape[1]):
        marker.points.append(
            Point(x=points[0, idx], y=points[1, idx], z=points[2, idx]))

    markers.markers.append(deepcopy(marker))

    return markers
