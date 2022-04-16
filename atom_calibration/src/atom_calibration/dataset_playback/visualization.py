"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

import copy
import math
import struct
from re import I

# 3rd-party
import cv2
import cv_bridge
import numpy as np

# import numpy as np  # TODO Eurico, line  fails if I don't do this
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
import tf2_ros
from atom_calibration.calibration.objective_function import *
from atom_calibration.collect.label_messages import *
from atom_core.cache import Cache
from atom_core.config_io import execute, readXacroFile, uriReader
from atom_core.dataset_io import (genCollectionPrefix, getCvImageFromDictionary, getCvImageFromDictionaryDepth,
                                  getPointCloudMessageFromDictionary)
from atom_core.image_processing import normalizeDepthImage
from atom_core.drawing import drawCross2D, drawSquare2D, drawLabelsOnImage, getRvizMarkersFrom3DLabels
from atom_core.naming import generateName
from atom_core.rospy_urdf_to_rviz_converter import urdfToMarkerArray
from colorama import Fore, Style
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, TransformStamped, Vector3
from matplotlib import cm
from rospy_message_converter import message_converter
from sensor_msgs.msg import PointCloud2, PointField, sensor_msgs
from std_msgs.msg import ColorRGBA, Header, UInt8MultiArray
from urdf_parser_py.urdf import URDF

# stdlib
from visualization_msgs.msg import Marker, MarkerArray

# own packages

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


def getCvImageFromCollectionSensorNonCached(collection_key, sensor_key, dataset):
    dictionary = dataset['collections'][collection_key]['data'][sensor_key]
    return getCvImageFromDictionary(dictionary)


def getCvDepthImageFromCollectionSensor(collection_key, sensor_key, dataset, scale=1000.0):
    dictionary = dataset['collections'][collection_key]['data'][sensor_key]
    return getCvImageFromDictionaryDepth(dictionary, scale=scale)


def createPatternMarkers(frame_id, ns, collection_key, now, dataset, graphics):
    markers = MarkerArray()

    # Draw pattern frame lines_sampled (top, left, right, bottom)
    marker = Marker(header=Header(frame_id=frame_id, stamp=now),
                    ns=ns + '-frame_sampled', id=0, frame_locked=True,
                    type=Marker.CUBE_LIST, action=Marker.ADD, lifetime=rospy.Duration(
                        0),
                    pose=Pose(position=Point(x=0, y=0, z=0),
                              orientation=Quaternion(x=0, y=0, z=0, w=1)),
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
                    type=Marker.CUBE_LIST, action=Marker.ADD, lifetime=rospy.Duration(
                        0),
                    pose=Pose(position=Point(x=0, y=0, z=0),
                              orientation=Quaternion(x=0, y=0, z=0, w=1)),
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
                   type=Marker.MESH_RESOURCE, action=Marker.ADD, lifetime=rospy.Duration(
                       0),
                   pose=Pose(position=Point(x=0, y=0, z=0),
                             orientation=Quaternion(x=0, y=0, z=0, w=1)),
                   scale=Vector3(x=1.0, y=1.0, z=1.0),
                   color=ColorRGBA(r=1, g=1, b=1, a=1))

        mesh_file, _, _ = uriReader(
            dataset['calibration_config']['calibration_pattern']['mesh_file'])
        m.mesh_resource = 'file://' + mesh_file  # mesh_resource needs uri format
        m.mesh_use_embedded_materials = True
        markers.markers.append(m)

    return markers  # return markers


def setupVisualization(dataset, args, selected_collection_key):
    """
    Creates the necessary variables in a dictionary "dataset_graphics", which will be passed onto the visualization
    function
    """

    # Create a python dictionary that will contain all the visualization related information
    graphics = {'collections': {}, 'sensors': {},
                'pattern': {}, 'ros': {'sensors': {}}, 'args': args}

    # Parse xacro description file
    description_file, _, _ = uriReader(
        dataset['calibration_config']['description_file'])
    rospy.loginfo('Reading description file ' + description_file + '...')
    # TODO not sure this should be done because of the use_tfs functionality ...
    xml_robot = readXacroFile(description_file)

    # Initialize ROS stuff
    rospy.init_node("dataset_playback")
    # graphics['ros']['tf_broadcaster'] = tf.TransformBroadcaster()
    graphics['ros']['tf_broadcaster'] = tf2_ros.TransformBroadcaster()

    # Sleep a litle to make sure the time.now() returns a correct time.
    rospy.sleep(0.2)
    now = rospy.Time.now()

    graphics['ros']['publisher_models'] = rospy.Publisher(
        '~robot_meshes', MarkerArray, queue_size=0, latch=True)
    # Analyse xacro and figure out which transforms are static (always the same over the optimization), and which are
    # not. For fixed we will use a static transform publisher.
    # for collection_key, collection in dataset['collections'].items():
    #     for transform_key, transform in collection['transforms'].items():
    #         parent = transform['parent']
    #         child = transform['child']
    #
    #
    #         for sensor_key, sensor in dataset['sensors'].items(): # is the transformation being optimized?
    #             if joint.parent == sensor['calibration_parent'] and joint.child == sensor['calibration_child']:
    #
    #
    #
    #         for joint in xml_robot.joints:
    #             if joint.parent == parent and joint.child == child:
    #                 print(transform)
    #                 print(joint)
    #
    #
    #
    #                 if joint.type == 'fixed':
    #                     transform['fixed'] = True
    #                 else:
    #                     transform['fixed'] = False

    # Create colormaps to be used for coloring the elements. Each collection contains a color, each sensor likewise.
    pattern = dataset['calibration_config']['calibration_pattern']
    graphics['pattern']['colormap'] = cm.gist_rainbow(
        np.linspace(0, 1, pattern['dimension']['x'] * pattern['dimension']['y']))

    # graphics['collections']['colormap'] = cm.tab20b(np.linspace(0, 1, len(dataset['collections'].keys())))
    graphics['collections']['colormap'] = cm.Pastel2(
        np.linspace(0, 1, len(dataset['collections'].keys())))
    for idx, collection_key in enumerate(sorted(dataset['collections'].keys())):
        graphics['collections'][str(collection_key)] = {
            'color': graphics['collections']['colormap'][idx, :]}

    # color_map_sensors = cm.gist_rainbow(np.linspace(0, 1, len(dataset['sensors'].keys())))
    # for idx, sensor_key in enumerate(sorted(dataset['sensors'].keys())):
    #     dataset['sensors'][str(sensor_key)]['color'] = color_map_sensors[idx, :]

    # Create image publishers ----------------------------------------------------------
    # We need to republish a new image at every visualization
    for sensor_key, sensor in dataset['sensors'].items():
        if sensor['modality'] == 'rgb':
            msg_type = sensor_msgs.msg.Image
            topic = dataset['calibration_config']['sensors'][sensor_key]['topic_name']
            topic_name = topic + '/labeled'
            graphics['collections'][str(sensor_key)] = {'publisher': rospy.Publisher(
                topic_name, msg_type, queue_size=0, latch=True)}
            msg_type = sensor_msgs.msg.CameraInfo
            topic_name = str(sensor_key) + '/camera_info'
            graphics['collections'][str(sensor_key)]['publisher_camera_info'] = \
                rospy.Publisher(topic_name, msg_type, queue_size=0, latch=True)

        if sensor['modality'] == 'depth':
            msg_type = sensor_msgs.msg.Image
            topic = dataset['calibration_config']['sensors'][sensor_key]['topic_name']
            topic_name = topic + '/labeled'
            graphics['collections'][str(sensor_key)] = {'publisher': rospy.Publisher(
                topic_name, msg_type, queue_size=0, latch=True)}
            msg_type = sensor_msgs.msg.CameraInfo
            topic_name = str(sensor_key) + '/camera_info'
            graphics['collections'][str(sensor_key)]['publisher_camera_info'] = \
                rospy.Publisher(topic_name, msg_type, queue_size=0, latch=True)

    # Create raw data publishers (non labeled data)--------------------------------------------
    for sensor_key, sensor in dataset['sensors'].items():
        graphics['ros']['sensors'][sensor_key] = {}
        graphics['ros']['sensors'][sensor_key]['collections'] = {}
        for collection_key, collection in dataset['collections'].items():
            if sensor['modality'] == 'lidar3d':

                frame_id = genCollectionPrefix(
                    collection_key, collection['data'][sensor_key]['header']['frame_id'])
                point_cloud_msg = getPointCloudMessageFromDictionary(
                    dataset['collections'][collection_key]['data'][sensor_key])

                point_cloud2 = PointCloud2(header=Header(frame_id=frame_id, stamp=now),
                                           height=point_cloud_msg.height,
                                           width=point_cloud_msg.width,
                                           fields=point_cloud_msg.fields,
                                           is_bigendian=point_cloud_msg.is_bigendian,
                                           point_step=point_cloud_msg.point_step,
                                           row_step=point_cloud_msg.row_step,
                                           data=point_cloud_msg.data,
                                           is_dense=point_cloud_msg.is_dense)

                graphics['ros']['sensors'][sensor_key]['collections'][collection_key] = {}
                graphics['ros']['sensors'][sensor_key]['collections'][collection_key]['PointCloud'] = point_cloud2
                graphics['ros']['sensors'][sensor_key]['collections'][collection_key]['PubPointCloud'] = rospy.Publisher(
                    str(sensor_key) + '/points', PointCloud2, queue_size=0, latch=True)

            if sensor['modality'] == 'depth':
                pass  # TODO add point cloud for depth sensors computed after depth image

    # Create Labeled Data publishers ----------------------------------------------------------
    markers = MarkerArray()
    for sensor_key, sensor in dataset['sensors'].items():
        for collection_key, collection in dataset['collections'].items():
            if not collection['labels'][str(sensor_key)]['detected']:  # not detected by sensor in collection
                continue

            if sensor['modality'] == 'lidar2d':  # -------- Publish the laser scan data ------------------------------
                frame_id = genCollectionPrefix(
                    collection_key, collection['data'][sensor_key]['header']['frame_id'])
                marker = Marker(header=Header(frame_id=frame_id, stamp=now),
                                ns=str(collection_key) + '-' + str(sensor_key), id=0, frame_locked=True,
                                type=Marker.POINTS, action=Marker.ADD, lifetime=rospy.Duration(
                                    0),
                                pose=Pose(position=Point(
                                    x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
                                scale=Vector3(x=0.03, y=0.03, z=0),
                                color=ColorRGBA(r=graphics['collections'][collection_key]['color'][0],
                                                g=graphics['collections'][collection_key]['color'][1],
                                                b=graphics['collections'][collection_key]['color'][2], a=1.0)
                                )

                # Get laser points that belong to the chessboard (labelled)
                idxs = collection['labels'][sensor_key]['idxs']
                rhos = [collection['data'][sensor_key]['ranges'][idx] for idx in idxs]
                thetas = [collection['data'][sensor_key]['angle_min'] +
                          collection['data'][sensor_key]['angle_increment'] * idx for idx in idxs]

                for idx, (rho, theta) in enumerate(zip(rhos, thetas)):
                    marker.points.append(
                        Point(x=rho * math.cos(theta), y=rho * math.sin(theta), z=0))

                markers.markers.append(copy.deepcopy(marker))

                # Draw extrema points
                marker.ns = str(collection_key) + '-' + str(sensor_key)
                marker.type = Marker.SPHERE_LIST
                marker.id = 1
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 0.5
                marker.points = [marker.points[0], marker.points[-1]]

                markers.markers.append(copy.deepcopy(marker))

                # Draw detected edges
                marker.ns = str(collection_key) + '-' + str(sensor_key)
                marker.type = Marker.CUBE_LIST
                marker.id = 2
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.a = 0.5

                marker.points = []  # Reset the list of marker points
                # add edge points
                for edge_idx in collection['labels'][sensor_key]['edge_idxs']:
                    p = Point()
                    p.x = rhos[edge_idx] * math.cos(thetas[edge_idx])
                    p.y = rhos[edge_idx] * math.sin(thetas[edge_idx])
                    p.z = 0
                    marker.points.append(p)
                markers.markers.append(copy.deepcopy(marker))

            if sensor['modality'] == 'lidar3d':  # -------- Publish the velodyne data ---------------------------

                topic = dataset['calibration_config']['sensors'][sensor_key]['topic_name']
                topic_name = topic + '/labeled'

                markers = getRvizMarkersFrom3DLabels(dataset, collection_key, sensor_key, now,
                                                     graphics['collections'][collection_key]['color'],
                                                     markers=markers, use_cache=False)

                graphics['ros']['sensors'][sensor_key]['collections'][collection_key]['MarkersLabeled'] = markers
                graphics['ros']['sensors'][sensor_key]['PubLabeled'] = rospy.Publisher(
                    topic_name, MarkerArray, queue_size=0, latch=True)

    # graphics['ros']['sensors'][sensor_key]['MarkersLabeled'] = markers
    # graphics['ros']['sensors'][sensor_key]['PubLabeled'] = rospy.Publisher(
    # '~' + sensor_key + '/labeled_data', MarkerArray, queue_size=0, latch=True)

    # -----------------------------------------------------------------------------------------------------
    # -------- Robot meshes
    # -----------------------------------------------------------------------------------------------------

    # Evaluate for each link if it may move or not (movable or immovalbe), to see if it needs to be drawn for each
    # collection. This is done by comparing the several transformations from the world_link to the <link> obtained
    # from the collections.
    immovable_links = []
    movable_links = []
    for link in xml_robot.links:  # cycle all links

        # print(dataset['calibration_config']
        #       ['world_link'] + ' to ' + link.name + ':')
        first_time = True
        for collection_key, collection in dataset['collections'].items():
            transform = atom_core.atom.getTransform(dataset['calibration_config']['world_link'], link.name,
                                                    collection['transforms'])
            if first_time:
                first_time = False
                transform_first_time = transform
            elif not np.array_equal(transform_first_time, transform):
                movable_links.append(link.name)
                break

        if link.name not in movable_links:
            immovable_links.append(link.name)

    # print('immovable links are: ' + str(immovable_links))
    # print('movable links are: ' + str(movable_links))

    # Check whether the robot is static, in the sense that all of its joints are fixed. If so, for efficiency purposes,
    # only one robot mesh (from the selected collection) is published.
    if args['all_joints_fixed']:  # assume the robot is static
        all_joints_fixed = True
        print('Robot is assumed to have all joints fixed.')
    else:  # run automatic detection
        all_joints_fixed = True
        for joint in xml_robot.joints:
            if not joint.type == 'fixed':
                print('Robot has at least joint ' + joint.name +
                      ' non fixed. Will render all collections')
                all_joints_fixed = False
                break

    markers = MarkerArray()
    if all_joints_fixed:  # render a single robot mesh
        print('Robot has all joints fixed. Will render only collection ' +
              selected_collection_key)
        rgba = [.5, .5, .5, 1]  # best color we could find
        m = urdfToMarkerArray(xml_robot, frame_id_prefix=genCollectionPrefix(selected_collection_key, ''),
                              namespace='immovable',
                              rgba=rgba)
        markers.markers.extend(m.markers)

    else:  # render robot meshes for all collections
        print('Robot has some dynamic joints. Will use advanced rendering ...')

        # Draw immovable links
        rgba = [.5, .5, .5, 1]  # best color we could find
        m = urdfToMarkerArray(xml_robot, frame_id_prefix=genCollectionPrefix(selected_collection_key, ''),
                              namespace='immovable',
                              rgba=rgba, skip_links=movable_links)
        markers.markers.extend(m.markers)

        # Draw movable links
        for collection_key, collection in dataset['collections'].items():
            rgba = graphics['collections'][collection_key]['color']
            rgba[3] = 0.2  # change the alpha
            m = urdfToMarkerArray(xml_robot, frame_id_prefix=genCollectionPrefix(collection_key, ''),
                                  namespace=collection_key,
                                  rgba=rgba, skip_links=immovable_links)
            markers.markers.extend(m.markers)

            # add a ghost (low alpha) robot marker at the initial pose
            if args['initial_pose_ghost']:
                rgba = [.1, .1, .8, 0.1]  # best color we could find
                # Draw immovable links
                m = urdfToMarkerArray(xml_robot, frame_id_prefix=genCollectionPrefix(collection_key, ''),
                                      frame_id_suffix=generateName(
                                          '', suffix='ini'),
                                      namespace=generateName(
                                          'immovabl', suffix='ini'),
                                      rgba=rgba, skip_links=movable_links)
                markers.markers.extend(m.markers)

                # Draw movable links
                for collection_key, collection in dataset['collections'].items():
                    m = urdfToMarkerArray(xml_robot, frame_id_prefix=genCollectionPrefix(collection_key, ''),
                                          frame_id_suffix=generateName(
                                              '', suffix='ini'),
                                          namespace=generateName(
                                              collection_key, suffix='ini'),
                                          rgba=rgba, skip_links=immovable_links)
                    markers.markers.extend(m.markers)

    graphics['ros']['RobotMeshMarkers'] = markers

    # -----------------------------------------------------------------------------------------------------
    # -------- Publish the pattern data
    # -----------------------------------------------------------------------------------------------------
    # Draw single pattern for selected collection key
    if dataset['calibration_config']['calibration_pattern']['fixed']:
        frame_id = generateName(dataset['calibration_config']['calibration_pattern']['link'],
                                prefix='c' + selected_collection_key)
        ns = str(selected_collection_key)
        markers = createPatternMarkers(
            frame_id, ns, selected_collection_key, now, dataset, graphics)
    else:  # Draw a pattern per collection
        markers = MarkerArray()
        for idx, (collection_key, collection) in enumerate(dataset['collections'].items()):
            frame_id = generateName(dataset['calibration_config']['calibration_pattern']['link'],
                                    prefix='c' + collection_key)
            ns = str(collection_key)
            collection_markers = createPatternMarkers(
                frame_id, ns, collection_key, now, dataset, graphics)
            markers.markers.extend(collection_markers.markers)

    graphics['ros']['MarkersPattern'] = markers
    graphics['ros']['PubPattern'] = rospy.Publisher(
        '~patterns', MarkerArray, queue_size=0, latch=True)

    # Create LaserBeams Publisher -----------------------------------------------------------
    # This one is recomputed every time in the objective function, so just create the generic properties.
    markers = MarkerArray()

    for collection_key, collection in dataset['collections'].items():
        for sensor_key, sensor in dataset['sensors'].items():
            # chess not detected by sensor in collection
            if not collection['labels'][sensor_key]['detected']:
                continue
            # if sensor['msg_type'] == 'LaserScan' or sensor['msg_type'] == 'PointCloud2':
            if sensor['modality'] == 'lidar2d' or sensor['modality'] == 'lidar3d':
                frame_id = genCollectionPrefix(
                    collection_key, collection['data'][sensor_key]['header']['frame_id'])
                marker = Marker(header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                                ns=str(collection_key) + '-' + str(sensor_key), id=0, frame_locked=True,
                                type=Marker.LINE_LIST, action=Marker.ADD, lifetime=rospy.Duration(
                                    0),
                                pose=Pose(position=Point(
                                    x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)),
                                scale=Vector3(x=0.001, y=0, z=0),
                                color=ColorRGBA(r=graphics['collections'][collection_key]['color'][0],
                                                g=graphics['collections'][collection_key]['color'][1],
                                                b=graphics['collections'][collection_key]['color'][2], a=1.0)
                                )
                markers.markers.append(marker)

    graphics['ros']['MarkersLaserBeams'] = markers
    graphics['ros']['PubLaserBeams'] = rospy.Publisher(
        '~LaserBeams', MarkerArray, queue_size=0, latch=True)

    # Create Miscellaneous MarkerArray -----------------------------------------------------------
    markers = MarkerArray()

    # Text signaling the anchored sensor
    for _sensor_key, sensor in dataset['sensors'].items():
        if _sensor_key == dataset['calibration_config']['anchored_sensor']:
            marker = Marker(header=Header(frame_id=str(_sensor_key), stamp=now),
                            ns=str(_sensor_key), id=0, frame_locked=True,
                            type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, lifetime=rospy.Duration(
                                0),
                            pose=Pose(position=Point(x=0, y=0, z=0.2),
                                      orientation=Quaternion(x=0, y=0, z=0, w=1)),
                            scale=Vector3(x=0.0, y=0.0, z=0.1),
                            color=ColorRGBA(r=0.6, g=0.6, b=0.6, a=1.0), text='Anchored')

            markers.markers.append(marker)

    graphics['ros']['MarkersMiscellaneous'] = markers
    graphics['ros']['PubMiscellaneous'] = rospy.Publisher(
        '~Miscellaneous', MarkerArray, queue_size=0, latch=True)
    # Publish only once in latched mode
    graphics['ros']['PubMiscellaneous'].publish(
        graphics['ros']['MarkersMiscellaneous'])
    graphics['ros']['Rate'] = rospy.Rate(10)
    graphics['ros']['Rate'] = rospy.Rate(10)
    # tfs need to be published at high frequencies. On the other hand, robot markers
    graphics['ros']['Counter'] = 0
    # should be published at low frequencies. This counter will serve to control this mechanism.

    return graphics


def visualizationFunction(models, selection, clicked_points=None):
    # print(Fore.RED + 'Visualization function called.' + Style.RESET_ALL)
    # Get the data from the meshes
    dataset = models['dataset']
    args = models['args']
    collections = models['dataset']['collections']
    sensors = models['dataset']['sensors']
    patterns = models['dataset']['patterns']
    config = models['dataset']['calibration_config']
    graphics = models['graphics']

    selected_collection_key = selection['collection_key']
    previous_selected_collection_key = selection['previous_collection_key']

    collection = dataset['collections'][selected_collection_key]

    # print("args['initial_pose_ghost'])" + str(args['initial_pose_ghost']))

    now = rospy.Time.now()  # time used to publish all visualization messages

    transfoms = []
    for collection_key, collection in collections.items():

        # To have a fully connected tree, must connect the instances of the tf tree of every collection into a single
        # tree. We do this by publishing an identity transform between the configured world link and hte world link
        # of each collection.
        parent = config['world_link']
        child = generateName(config['world_link'], prefix='c' + collection_key)

        transform = TransformStamped(header=Header(frame_id=parent, stamp=now),
                                     child_frame_id=child,
                                     transform=Transform(translation=Vector3(x=0, y=0, z=0),
                                                         rotation=Quaternion(x=0, y=0, z=0, w=1)))
        transfoms.append(transform)

        for transform_key, transform in collection['transforms'].items():
            parent = generateName(
                transform['parent'], prefix='c' + collection_key)
            child = generateName(
                transform['child'], prefix='c' + collection_key)
            x, y, z = transform['trans']
            qx, qy, qz, qw = transform['quat']
            transform = TransformStamped(header=Header(frame_id=parent, stamp=now),
                                         child_frame_id=child,
                                         transform=Transform(translation=Vector3(x=x, y=y, z=z),
                                                             rotation=Quaternion(x=qx, y=qy, z=qz, w=qw)))
            transfoms.append(transform)

    graphics['ros']['tf_broadcaster'].sendTransform(transfoms)

    # print("graphics['ros']['Counter'] = " + str(graphics['ros']['Counter']))
    if graphics['ros']['Counter'] < 5:
        graphics['ros']['Counter'] += 1
        return None
    else:
        graphics['ros']['Counter'] = 0

    # Update markers stamp, so that rviz uses newer transforms to compute their poses.
    for marker in graphics['ros']['RobotMeshMarkers'].markers:
        marker.header.stamp = now

    # Publish the meshes
    graphics['ros']['publisher_models'].publish(graphics['ros']['RobotMeshMarkers'])

    # Update timestamp for the patterns markers
    for marker in graphics['ros']['MarkersPattern'].markers:
        marker.header.stamp = now

    # Update timestamp for labeled markers (for lidar3d and depth)
    for sensor_key in graphics['ros']['sensors']:
        if not dataset['sensors'][sensor_key]['modality'] == 'lidar3d':
            continue

        markers = MarkerArray()
        markers = getRvizMarkersFrom3DLabels(dataset, collection_key, sensor_key, now,
                                             graphics['collections'][selected_collection_key]['color'],
                                             markers=markers, use_cache=False)

        graphics['ros']['sensors'][sensor_key]['PubLabeled'].publish(markers)

    # Update timestamp for laser beams markers
    for marker in graphics['ros']['MarkersLaserBeams'].markers:
        marker.header.stamp = now

    # Update timestamp and publish raw data pointcloud2 message
    for sensor_key in graphics['ros']['sensors']:
        if not dataset['sensors'][sensor_key]['modality'] == 'lidar3d':
            continue

        point_cloud_msg = graphics['ros']['sensors'][sensor_key]['collections'][selected_collection_key]['PointCloud']
        point_cloud_msg.header.stamp = now
        graphics['ros']['sensors'][sensor_key]['collections'][selected_collection_key]['PubPointCloud'].publish(
            point_cloud_msg)

    # Create a new marker array which contains only the marker related to the selected collection
    # Publish the pattern data
    marker_array_1 = MarkerArray()
    for marker in graphics['ros']['MarkersPattern'].markers:
        prefix = marker.header.frame_id[:3]
        if prefix == 'c' + str(selected_collection_key) + '_':
            marker_array_1.markers.append(marker)
            marker_array_1.markers[-1].action = Marker.ADD
        elif not previous_selected_collection_key == selected_collection_key and prefix == 'c' + str(
                previous_selected_collection_key) + '_':
            marker_array_1.markers.append(marker)
            marker_array_1.markers[-1].action = Marker.DELETE
    graphics['ros']['PubPattern'].publish(marker_array_1)

    # Create a new marker array which contains only the marker related to the selected collection
    # Publish the robot_mesh_
    marker_array_2 = MarkerArray()
    for marker in graphics['ros']['RobotMeshMarkers'].markers:
        prefix = marker.header.frame_id[:3]
        if prefix == 'c' + str(selected_collection_key) + '_':
            marker_array_2.markers.append(marker)
            marker_array_2.markers[-1].action = Marker.ADD
        elif not previous_selected_collection_key == selected_collection_key and prefix == 'c' + str(
                previous_selected_collection_key) + '_':
            marker_array_2.markers.append(marker)
            marker_array_2.markers[-1].action = Marker.DELETE
    graphics['ros']['publisher_models'].publish(
        graphics['ros']['RobotMeshMarkers'])

    # Create a new marker array which contains only the marker related to the selected collection
    # Publish the robot_mesh_
    marker_array_3 = MarkerArray()
    for marker in graphics['ros']['MarkersLaserBeams'].markers:
        prefix = marker.header.frame_id[:3]
        if prefix == 'c' + str(selected_collection_key) + '_':
            marker_array_3.markers.append(marker)
            marker_array_3.markers[-1].action = Marker.ADD
        elif not previous_selected_collection_key == selected_collection_key and prefix == 'c' + str(
                previous_selected_collection_key) + '_':
            marker_array_3.markers.append(marker)
            marker_array_3.markers[-1].action = Marker.DELETE

    graphics['ros']['PubLaserBeams'].publish(
        graphics['ros']['MarkersLaserBeams'])

    # Publish Annotated images
    for sensor_key, sensor in sensors.items():
        # if sensor['msg_type'] == 'Image':
        if sensor['modality'] == 'rgb':
            if args['show_images']:
                collection = dataset['collections'][selected_collection_key]

                # print(collection['labels'][sensor_key].keys())

                image = copy.deepcopy(getCvImageFromCollectionSensorNonCached(
                    selected_collection_key, sensor_key, dataset))
                width = collection['data'][sensor_key]['width']
                height = collection['data'][sensor_key]['height']
                diagonal = math.sqrt(width ** 2 + height ** 2)
                cm = graphics['pattern']['colormap']

                # Draw ground truth points (as squares)
                for idx, point in enumerate(collection['labels'][sensor_key]['idxs']):
                    # print(point)
                    x = int(round(point['x']))
                    y = int(round(point['y']))
                    color = (cm[idx, 2] * 255, cm[idx, 1]
                             * 255, cm[idx, 0] * 255)
                    drawSquare2D(image, x, y, int(8E-3 * diagonal),
                                 color=color, thickness=2)

                msg = CvBridge().cv2_to_imgmsg(image, "bgr8")
                msg.header.frame_id = 'c' + selected_collection_key + '_' + sensor['parent']
                graphics['collections'][sensor_key]['publisher'].publish(msg)

                # Publish camera info message
                camera_info_msg = message_converter.convert_dictionary_to_ros_message(
                    'sensor_msgs/CameraInfo', sensor['camera_info'])
                camera_info_msg.header.frame_id = msg.header.frame_id
                graphics['collections'][sensor_key]['publisher_camera_info'].publish(
                    camera_info_msg)

        if sensor['modality'] == 'depth':
            if args['show_images']:

                # Shortcut variables
                collection = collections[selected_collection_key]
                clicked_sensor_points = clicked_points[selected_collection_key][sensor_key]['points']

                # Create image to draw on top
                image = getCvImageFromDictionaryDepth(collection['data'][sensor_key])
                gui_image = normalizeDepthImage(image, max_value=5)
                gui_image = drawLabelsOnImage(collection['labels'][sensor_key], gui_image)

                if not clicked_points[selected_collection_key][sensor_key]['valid_polygon']:
                    # Draw a cross for each point
                    for point in clicked_sensor_points:
                        drawSquare2D(gui_image, point['x'], point['y'], size=5, color=(50, 190, 0))

                    # Draw a line segment for each pair of consecutive points
                    for point_start, point_end in zip(clicked_sensor_points[:-1], clicked_sensor_points[1:]):
                        cv2.line(gui_image, pt1=(point_start['x'], point_start['y']),
                                 pt2=(point_end['x'], point_end['y']),
                                 color=(0, 0, 255), thickness=1)

                msg = CvBridge().cv2_to_imgmsg(gui_image, "passthrough")

                msg.header.frame_id = 'c' + selected_collection_key + '_' + sensor['parent']
                graphics['collections'][sensor_key]['publisher'].publish(msg)

                # Publish camera info message
                camera_info_msg = message_converter.convert_dictionary_to_ros_message('sensor_msgs/CameraInfo',
                                                                                      sensor['camera_info'])
                camera_info_msg.header.frame_id = msg.header.frame_id
                graphics['collections'][sensor_key]['publisher_camera_info'].publish(
                    camera_info_msg)

    graphics['ros']['Rate'].sleep()
