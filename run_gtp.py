#!/usr/local/bin/python



from __future__ import print_function
import yaml
from sys import argv
from robot.mxnet_robot import MXNetRobot

from betago.gtp import GTPFrontend
from data_loader.original_processor import OriginalProcessor


checkpoint_file = argv[1]
epoch = int(argv[2])

processor = OriginalProcessor('data/standard/streight_and_curve15.sgf')
bot = MXNetRobot(checkpoint_file, epoch, processor)


frontend = GTPFrontend(bot=bot)

frontend.run()
