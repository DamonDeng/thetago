class Point(object):

  def __init__(self, color, pos):
    self.color = color # 0: empty, 1:black, 2:white
    self.pos = pos
    self.left = None
    self.right = None
    self.up = None
    self.down = None
    self.group_id = -1
  
  def get_color(self):
    return self.color

  def get_pos(self):
    return self.pos

  def set_group_id(self, id):
    self.group_id = id

  def set_color(self, color):
    self.color = color
  
  def get_color(self):
    return self.color

  def get_group_id(self):
    return self.group_id

  def set_left(self, left_point):
    self.left = left_point
  
  def get_left(self):
    return self.left

  def set_right(self, right_point):
    self.right = right_point
  
  def get_right(self):
    return self.right

  def set_up(self, up_point):
    self.up = up_point
  
  def get_up(self):
    return self.up

  def set_down(self, down_point):
    self.down = down_point
  
  def get_down(self):
    return self.down