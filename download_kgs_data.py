from data_loader.index_processor import KGSIndex
from data_loader.kgs_downloader import KGSDownloader

from gosgf import Sgf_game

import gzip
import shutil
import tarfile
import os
import os.path

print('Started to download KGS data...')

downloader = KGSDownloader()

downloader.retrieve_files()