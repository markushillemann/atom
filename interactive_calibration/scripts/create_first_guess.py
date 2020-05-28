#!/usr/bin/env python

import sys
import os.path
import argparse
from colorama import Fore, Style
import yaml

# ROS imports
import rospkg
import rospy

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler

from urdf_parser_py.urdf import URDF

# __self__ imports
from interactive_calibration.utilities import loadConfig
from interactive_calibration.sensor import Sensor


class InteractiveFirstGuess(object):

    def __init__(self, args):
        # command line arguments
        self.args = args
        # calibration config
        self.config = None
        # sensors
        self.sensors = []
        # robot URDF
        self.urdf = None
        # interactive markers server
        self.server = None
        self.menu = None

    def init(self):
        # The parameter /robot_description must exist
        if not rospy.has_param('/robot_description'):
            rospy.logerr("Parameter '/robot_description' must exist to continue")
            sys.exit(1)
        # Load the urdf from /robot_description
        self.urdf = URDF.from_parameter_server()

        self.config = loadConfig(self.args['calibration_file'])
        if self.config is None:
            sys.exit(1)  # loadJSON should tell you why.

        # ok = validateLinks(self.config.world_link, self.config.sensors, self.urdf)
        # if not ok:
        #     sys.exit(1)

        print('Number of sensors: ' + str(len(self.config['sensors'])))

        # Init interaction
        self.server = InteractiveMarkerServer('first_guess_node')
        self.menu = MenuHandler()

        self.menu.insert("Save sensors configuration", callback=self.onSaveFirstGuess)
        self.menu.insert("Reset to initial configuration", callback=self.onReset)

        # For each node generate an interactive marker.
        for name, sensor in self.config['sensors'].items():
            print(Fore.BLUE + '\nSensor name is ' + name + Style.RESET_ALL)
            params = {
                "frame_world": self.config['world_link'],
                "frame_opt_parent": sensor['parent_link'],
                "frame_opt_child": sensor['child_link'],
                "frame_sensor": sensor['link'],
                "marker_scale": self.args['marker_scale']}
            # Append to the list of sensors
            self.sensors.append(Sensor(name, self.server, self.menu, **params))

        self.server.applyChanges()

    def onSaveFirstGuess(self, feedback):
        for sensor in self.sensors:
            # find corresponding joint for this sensor
            for joint in self.urdf.joints:
                if sensor.opt_child_link == joint.child and sensor.opt_parent_link == joint.parent:
                    trans = sensor.optT.getTranslation()
                    euler = sensor.optT.getEulerAngles()
                    joint.origin.xyz = list(trans)
                    joint.origin.rpy = list(euler)

        # Write the urdf file with interactive_calibration's
        # source path as base directory.
        rospack = rospkg.RosPack()
        # outfile = rospack.get_path('interactive_calibration') + os.path.abspath('/' + self.args['filename'])
        outfile = self.args['filename']
        with open(outfile, 'w') as out:
            print("Writing fist guess urdf to '{}'".format(outfile))
            out.write(self.urdf.to_xml_string())

    def onReset(self, feedback):
        for sensor in self.sensors:
            sensor.resetToInitalPose()


if __name__ == "__main__":

    # Parse command line arguments
    ap = argparse.ArgumentParser(description='Create first guess. Sets up rviz interactive markers that allow the '
                                             'user to define the position and orientation of each of the sensors '
                                             'listed in the config.yml.')
    ap.add_argument("-f", "--filename", type=str, required=True, default="/calibrations/atlascar2"
                                                                         "/atlascar2_first_guess.urdf.xacro",
                    help="Full path and name of the first guess xacro file. Starting from the root of the interactive "
                         "calibration ros package")
    ap.add_argument("-s", "--marker_scale", type=float, default=0.6, help='Scale of the interactive markers.')
    ap.add_argument("-c", "--calibration_file", type=str, required=True, help='full path to calibration file.')

    # Roslaunch files send a "__name:=..." argument (and __log:=) which disrupts the argparser. The solution is to
    # filter this argv. in addition, the first argument is the node name, which should also not be given to the
    # parser.
    argv_filtered = []
    for i, argv in enumerate(sys.argv):
        if (not all(x in argv for x in ['__', ':='])) and (i != 0):
            argv_filtered.append(argv)
    # print('\n' + str(sys.argv))
    # print('\n' + str(argv_filtered))
    args = vars(ap.parse_args(args=argv_filtered))

    # Initialize ROS stuff
    rospy.init_node("first_guess_node")

    # Launch the application !!
    first_guess = InteractiveFirstGuess(args)
    first_guess.init()

    print('Changes applied ...')

    rospy.spin()
