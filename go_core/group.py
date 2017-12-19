 
  
class Group(object):

  def __init__(self, group_id, group_type):
    
    self.empty_stack = set()
    self.black_stack = set()
    self.white_stack = set()
    self.group_id = group_id
    self.group_type = group_type # group type: 0 empty group; 1 black group; 2 white group 

  def get_group_id(self):
    return self.group_id

  def get_group_type(self):
    return self.group_type

  def get_score(self):
    empty_number = self.get_empty_number()
    black_number = self.get_black_number()
    white_number = self.get_white_number()

    if self.group_type == 1:
      # black group
      return (0, black_number, 0)
    elif self.group_type == 2:
      # white group
      return (0, 0, white_number)
    else:
      # empty group
      if black_number > 0 and white_number > 0:
        # this empty group doesn't belong to any color
        return (empty_number, 0, 0)
      elif black_number == 0 and white_number == 0:
        # init state, no black stone, no white stone either
        return (empty_number, 0, 0)
      elif black_number == 1 and white_number == 0:
        # first step, only one stone in the pane
        return (empty_number, 0, 0)
      else:
        if black_number > 0 and white_number == 0:
          # this empty group belong to black
          return (0, empty_number, 0)
        elif white_number > 0 and black_number == 0:
          # this empty group belong to white
          return (0, 0, empty_number)
        else:
          # this function will end up here is there is anything inconsistent
          # return (0, 0, 0) for this group so that we can detect the inconsistent status out side
          return (0, 0, 0)

  def add_empty(self, pos):
    self.empty_stack.add(pos)

  def remove_empty(self, color_value, pos):
    if pos in self.empty_stack:
      self.empty_stack.remove(pos)

    if color_value == 1:
      self.black_stack.add(pos)
    elif color_value == 2:
      self.white_stack.add(pos)

  def remove_color(self, color_value, pos):
    if color_value == 1:
      if pos in self.black_stack:
        self.black_stack.remove(pos)
    
    if color_value == 2:
      if pos in self.white_stack:
        self.white_stack.remove(pos)

    self.empty_stack.add(pos)


  def add_point(self, color_value, pos):
    if color_value == 1:
      self.black_stack.add(pos)
    elif color_value == 2:
      self.white_stack.add(pos)

  def add_black(self, pos):
    self.black_stack.add(pos)

  def add_white(self, pos):
    self.white_stack.add(pos)

  def empty_stack(self):
    return self.empty_stack

  def black_stack(self):
    return self.black_stack

  def white_stack(self):
    return self.white_stack

  def get_stack(self, color_value):
    if color_value == 0:
      return self.empty_stack
    elif color_value == 1:
      return self.black_stack
    elif color_value == 2:
      return self.white_stack

  def get_stack_len(self, color_value):
    if color_value == 0:
      return len(self.empty_stack)
    elif color_value == 1:
      return len(self.black_stack)
    elif color_value == 2:
      return len(self.white_stack)

  def get_empty_number(self):
    return len(self.empty_stack)

  def get_black_number(self):
    return len(self.black_stack)

  def get_white_number(self):
    return len(self.white_stack)

  def is_the_only_empty(self, cur_pos):
    # return true, if cur_pos is the only empty point in current empty stack
    result = False
    if self.get_empty_number() == 1:
      if cur_pos in self.empty_stack:
        result = True
    return result

  def print_debug_info(self):
    print ('# group_id:' + str(self.group_id)),
    print (' group_type:' + str(self.group_type)),
    print (' Empty:' + str(self.get_empty_number())),
    print (' Black:' + str(self.get_black_number())),
    print (' White:' + str(self.get_white_number()))
    
  