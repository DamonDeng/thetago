#!/usr/local/bin/python



from __future__ import print_function
import yaml
from sys import argv
from robot.mxnet_robot import MXNetRobot

from gtp import GTPFrontend
from data_loader.original_processor import OriginalProcessor


checkpoint_file = argv[1]
epoch = int(argv[2])

if len(argv) > 3:
  value_file = argv[3]
  value_epoch = int(argv[4])
else:
  value_file = None
  value_epoch = None

processor = OriginalProcessor
bot = MXNetRobot(checkpoint_file, epoch, processor, value_file, value_epoch)


frontend = GTPFrontend(bot=bot)

frontend.run()
