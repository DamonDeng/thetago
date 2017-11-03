
import sys
sys.path.append("..")

from .index_processor import KGSIndex
from gosgf import *

import gzip
import shutil
import tarfile
import os
import os.path
import random

class KGSDownloader(object):

  def __init__(self,
                 data_directory='data',
                 train_directory='data/train',
                 validate_directory='data/validate',
                 test_directory='data/test',
                 validate_percentage=20,
                 test_percentage=20):
        '''
        Download Go Games on KGS, extract them into sgf files into train,validate,test directory.

        Parameters:
        -----------
        data_directory: name of directory relative to current path to store SGF data
        train_directory: name of directory to store training SGF data
        validate_directory: name of directory to store validating SGF data
        test_directory: name of directory to store testing SGF data
        validate_percentage: how many SFG files to be extracted to validating directory
        test_percentage: how many SFG files to be extracted to testing directory
        '''
        
        self.data_directory = data_directory
        self.train_directory = train_directory
        self.validate_directory = validate_directory
        self.test_directory = test_directory
        self.validate_percentage = validate_percentage
        self.test_percentage = test_percentage
  
  def download_files(self):
    index = KGSIndex(data_directory=self.data_directory)
    index.download_files()
    return index

  def retrieve_files(self):
    index = self.download_files()

    for file_info in index.file_info:
        print(str(file_info))
        filename = file_info['filename']
        self.extract_zip(self.data_directory, filename)
        
  def extract_zip(self, dir_name, zip_file_name):
    this_gz = gzip.open(dir_name + '/' + zip_file_name)
    this_tar_file = zip_file_name[0:-3]
    this_tar = open(dir_name + '/' + this_tar_file, 'wb')
    shutil.copyfileobj(this_gz, this_tar)  # random access needed to tar
    this_tar.close()
    this_zip = tarfile.open(dir_name + '/' + this_tar_file)
    name_list = this_zip.getnames()
    
    total_number = len(name_list)
    test_number = int(total_number * self.test_percentage /100)
    validate_number = int(total_number * self.validate_percentage/100)

    test_name_list = random.sample(name_list, test_number)
    other_left = list(set(name_list).difference(set(test_name_list)))
    
    validate_name_list = random.sample(other_left, validate_number)

    train_name_list = list(set(other_left).difference(set(validate_name_list)))

    self.extract_sgf(this_zip, test_name_list, self.test_directory)
    self.extract_sgf(this_zip, validate_name_list, self.validate_directory)
    self.extract_sgf(this_zip, train_name_list, self.train_directory)
    

    # print ('Total:' + str(len(name_list)))
    # print ('Len of test:' + str(len(test_name_list)))
    # print ('Len of val:' + str(len(validate_name_list)))
    # print ('Len of train:' + str(len(train_name_list)))
    
    
    # print(name_list[4])
    # read_sgf_from_zip(this_zip, name_list[4])

  def extract_sgf(self, this_zip, name_list, target_dir):
    
    if not os.path.exists(target_dir):
      os.makedirs(target_dir)

    for name in name_list:
      if name.endswith('.sgf'):
        sgf_content = this_zip.extractfile(name).read()
        open(target_dir+'/'+name.split('/')[-1],'w').write(sgf_content)
        # print ('Dry run: ' + target_dir+'/'+name.split('/')[-1])

    