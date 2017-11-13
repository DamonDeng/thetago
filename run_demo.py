#!/usr/bin/env python
from __future__ import print_function
import argparse
import yaml
import os
import webbrowser

from robot.server import HTTPFrontend
from robot.mxnet_robot import MXNetRobot


parser = argparse.ArgumentParser()
parser.add_argument('--host', default='localhost', help='host to listen to')
parser.add_argument('--port', '-p', type=int, default=8080,
                    help='Port the web server should listen on (default 8080).')
args = parser.parse_args()

# Open web frontend and serve model
webbrowser.open('http://{}:{}/'.format(args.host, args.port), new=2)
go_robot = MXNetRobot('model_zoo/thetago_standard', 100)
go_server = HTTPFrontend(bot=go_robot, port=args.port)
go_server.run()
