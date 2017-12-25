 
  
class Group(object):

  def __init__(self, group_id, group_type, first_group=False):
    
    self.empty_stack = set()
    self.black_stack = set()
    self.white_stack = set()
    self.group_id = group_id
    self.group_type = group_type # group type: 0 empty group; 1 black group; 2 white group 
    self.neighbour_groups = set()
    self.empty_neighbour_groups = set()
    self.black_neighbour_groups = set()
    self.white_neighbour_groups = set()
    
    self.first_group = first_group

  def get_group_id(self):
    return self.group_id

  def get_group_type(self):
    return self.group_type

  def add_neighbour(self, target_group):

    # print ('# adding neighbour group: type:' + str(target_group.get_group_type()))
    # print ('# current group:' + str(self.group_id) + '   target group:' + str(target_group.get_group_id()))
    self.neighbour_groups.add(target_group)
    if target_group.get_group_type() == 0:
      # print ('# adding empty group neighbour')
      self.empty_neighbour_groups.add(target_group)
    elif target_group.get_group_type() == 1:
      # print ('# adding black group neighbour')
      self.black_neighbour_groups.add(target_group)
    elif target_group.get_group_type() == 2:
      # print ('# adding white group neighbour')
      self.white_neighbour_groups.add(target_group)

  def clear_neighbour(self):
    self.neighbour_groups = set()
    self.empty_neighbour_groups = set()
    self.black_neighbour_groups = set()
    self.white_neighbour_groups = set()

  def get_empty_neighbours(self):
    return self.empty_neighbour_groups

  def get_black_neighbours(self):
    return self.black_neighbour_groups

  def get_white_neighbours(self):
    return self.white_neighbour_groups

  def is_black_eye(self):
    # print ('# trying to check black eye')
    if self.is_black_space():
      # print ('# is black space')
      black_neighbour_number = len(self.black_neighbour_groups)
      # print ('# black neighbour number: ' + str(black_neighbour_number))
      if black_neighbour_number < 1:
        # inconsist state
        print('# warning: inconsist states, black space has no black neighbour')
      elif black_neighbour_number == 1:
        # there is only one group around this black space, it is try eye
        return True
      elif black_neighbour_number == 2:
        # there are two group around this black space, 
        # need to check whether these two have other common black space neighbour
        # print ('# found space with two black neighbour')
        neighbour_list = list(self.black_neighbour_groups)

        neighbour1 = neighbour_list[0]
        neighbour2 = neighbour_list[1]

        further_space_groups= neighbour1.get_empty_neighbours()
        further_eye1 = set()
        for further_space_group in further_space_groups:
          space_group_id = further_space_group.get_group_id()
          # print ('# further space group id:' + str(space_group_id))
          if space_group_id != self.group_id:
            if further_space_group.is_black_space():
              # print ('# further space is black space:' + str(space_group_id) + '   current id:' + str(self.group_id))
              further_eye1.add(space_group_id)

        if len(further_eye1) >= 1:
          # print('# has further space')
          further_space_groups = neighbour2.get_empty_neighbours()
          
          for further_space_group in further_space_groups:
            space_group_id = further_space_group.get_group_id()
            if space_group_id != self.group_id:            
              if further_space_group.is_black_space():
                if space_group_id in further_eye1:
                  return True
      else:
        # there are more than 2 group around this balck space
        # it is still possible that this space are true eye, 
        # but it is too complex to check it right now,
        # will do it when we have time, probably with some other clever way
        return False
    
    return False

  def is_white_eye(self):
    if self.is_white_space():
      white_neighbour_number = len(self.white_neighbour_groups)
      if white_neighbour_number < 1:
        # inconsist state
        print('# warning: inconsist states, white space has no white neighbour')
      elif white_neighbour_number == 1:
        # there is only one group around this white space, it is try eye
        return True
      elif white_neighbour_number == 2:
        # there are two group around this white space, 
        # need to check whether these two have other common black space neighbour
        
        neighbour_list = list(self.white_neighbour_groups)

        neighbour1 = neighbour_list[0]
        neighbour2 = neighbour_list[1]

        further_space_groups= neighbour1.get_empty_neighbours()
        further_eye1 = set()
        for further_space_group in further_space_groups:
          space_group_id = further_space_group.get_group_id()
          if space_group_id != self.group_id:
            if further_space_group.is_white_space():
              further_eye1.add(space_group_id)

        if len(further_eye1) >= 1:
          further_space_groups = neighbour2.get_empty_neighbours()
          
          for further_space_group in further_space_groups:
            space_group_id = further_space_group.get_group_id()
            if space_group_id != self.group_id:
              
              if further_space_group.is_white_space():
                if space_group_id in further_eye1:
                  return True
      else:
        # there are more than 2 group around this balck space
        # it is still possible that this space are true eye, 
        # but it is too complex to check it right now,
        # will do it when we have time, probably with some other clever way
        return False
    
    return False


  def is_black_space(self):
    result = False
    # it is a hard fix, to make sure that large size of space is not counted as any player's space
    max_space_number = 350
    if self.group_type == 0:
      if self.get_empty_number() < max_space_number:
        if self.get_white_number() == 0:
          if self.get_black_number() > 0:
            if not self.first_group:
              result = True
    return result

  def is_white_space(self):
    result = False
    # it is a hard fix, to make sure that large size of space is not counted as any player's space
    max_space_number = 350
    if self.group_type == 0:
      if self.get_empty_number() < max_space_number:
        if self.get_black_number() == 0:
          if self.get_white_number() > 0:
            if not self.first_group:
              result = True
    return result

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
    
  