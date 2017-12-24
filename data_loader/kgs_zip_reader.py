from gosgf import *

import gzip
import shutil
import tarfile
import os
import os.path

class KGSZipReader(object):

  def __init__(self, data_directory='data', file_limit=-1):
    self.data_directory = data_directory
    self.file_limit = file_limit

  def get_generator(self):

    number_of_file = 0
    for file in os.listdir(self.data_directory):  
      if self.file_limit > 0 and number_of_file > self.file_limit:
          break  
      number_of_file+=1
      file_path = os.path.join(self.data_directory, file)  
      if os.path.splitext(file_path)[1]=='.tar':  
          this_zip = tarfile.open(file_path)
          name_list = this_zip.getnames()
          for name in name_list:
            if name.endswith('.sgf'):
              sgf_content = this_zip.extractfile(name).read()
              yield (file_path, sgf_content)

  

    
    
    
