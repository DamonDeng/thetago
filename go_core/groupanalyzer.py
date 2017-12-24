import numpy as np

class GroupAnalyzer(object):

  def __init__(self, group_records, valid_ids, board_size):
    self.group_records = group_records
    self.valid_ids = valid_ids
    self.board_size = board_size
    self.result = np.zeros((board_size, board_size))
    self.history = set()

  def analyze(self):
    for group_id in self.valid_ids:
      cur_group = self.group_records.get(group_id)
      if cur_group.get_group_type() == 0:
        return 'todo'
    return self.result
