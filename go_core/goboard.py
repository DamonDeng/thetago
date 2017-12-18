import numpy as np
import copy

class GoBoard(object):

  def __init__(self, board_size=19):
    self.reset(board_size)

  def reset(self, board_size):        
    self.history_length = 2
    self.pane_number = 8

    # states attribute will keep all the states panes 
    # shpae of states: (history_length, pane_number, board_size, board_size)
    # pane_index:
    # 0: color: 0 for empty, 1 for black, 2 for white
    # 1: group_empty: number of empty current point has
    # 2: group_black: number of black current ponit has
    # 3: group_white: number of white current point has
    # 4: 
    self.states = np.zeros((self.history_length, self.pane_number, board_size, board_size))
    self.history_index = 0

    
    self.board_size = board_size

    self.group_reporters = {}

    self.group_id_index = 0 # the first group id is 0, which is used in comming initing code.

    self.all_points = {}

    for row in range(self.board_size):
      for col in range(self.board_size):
        cur_point = Point(0, (row,col))
        self.all_points[(row, col)] = cur_point
    
    self.connect_all_points()

    empty_color = 0
    working_group_id = 0
    point = self.all_points.get((0,0))
    reporter = Reporter(working_group_id, empty_color)
    reporter = self.update_group(point, empty_color, working_group_id, reporter)
    # print ('# update finished, reporter empty number: ' + str(reporter.get_empty_number()))
    self.group_reporters[working_group_id] = reporter

    self.last_move = None # last move format: (pos, color_value)
    self.last_remove = set() # init the last remove stones , it is a set.



  def apply_move(self, color, pos):
    # apply move to position
    print('# applying move: ' + color + "  " + str(pos))
    cur_point = self.all_points.get(pos)
    if cur_point is None:
      # incorrect position, return
      return

    if cur_point.get_color() != 0:
      # not empty, return
      return
    
    up_point = cur_point.get_up()
    down_point = cur_point.get_down()
    left_point = cur_point.get_left()
    right_point = cur_point.get_right()

    # get color value and enemy color value of current point
    cur_color_value = self.get_color_value(color)
    enemy_color_value = self.get_enemy_color_value(color)

    # check whether the move is suicide
    if self.is_suicide(cur_color_value, pos):
      print('# warning: suicide move detected!!')
      return

    if self.is_ko(cur_color_value, pos):
      print('# warning: ko move detected!!')
      return
    
    # setting the color of current point.
    cur_point.set_color(cur_color_value)

    self.last_move = (pos, cur_color_value) # remember last move
    self.last_remove = set() # clear all the elements in last remove, so that we can add new ones


    # get point group id for current point and set to current point
    point_group_id = self.get_group_id_index()
    # cur_point.set_group_id(point_group_id)
    # update the point around if they have the same color with current point
    reporter = Reporter(point_group_id, cur_color_value)
    # add current point into the new group at first
    # reporter.add_point(cur_color_value, pos)

    # try to merge the friend point around
    # reporter = self.handle_friend_point(up_point, cur_color_value, point_group_id, reporter)
    # reporter = self.handle_friend_point(down_point, cur_color_value, point_group_id, reporter)
    # reporter = self.handle_friend_point(left_point, cur_color_value, point_group_id, reporter)
    # reporter = self.handle_friend_point(right_point, cur_color_value, point_group_id, reporter)
    # save the reporter of current point

    reporter = self.update_group(cur_point, cur_color_value, point_group_id, reporter)
    self.group_reporters[point_group_id] = reporter
    # reporter.print_debug_info()
    
    # remove the enemy aound if they have only one empty point
    # and current point is the only empty point they have
    # decrese empty point number if it is not the only empty point they have
    self.handle_enemy_point(up_point, enemy_color_value, cur_color_value, pos)
    self.handle_enemy_point(down_point, enemy_color_value, cur_color_value, pos)
    self.handle_enemy_point(left_point, enemy_color_value, cur_color_value, pos)
    self.handle_enemy_point(right_point, enemy_color_value, cur_color_value, pos)
  
    # ---------------------------------
    # update the empty space group

    # normal group id start from 0
    # for the point where stones are just removed, the group id is -1
    # setting working_group_id to -2, so that it wouldn't be any group id is equal to the init group id
    # working_group_id = -2
    working_group_id_set = set()
    # working_group_id_set.add(working_group_id)
    
    working_group_id_set = self.handle_space_point(up_point, working_group_id_set)
    working_group_id_set = self.handle_space_point(down_point, working_group_id_set)
    working_group_id_set = self.handle_space_point(left_point, working_group_id_set)
    working_group_id_set = self.handle_space_point(right_point, working_group_id_set) 

  def handle_friend_point(self, point, cur_color_value, point_group_id, reporter):
    if point != None and point.get_color() == cur_color_value:  
        reporter = self.update_group(point, cur_color_value, point_group_id, reporter)
    return reporter

  def handle_enemy_point(self, point, enemy_color_value, cur_color_value, cur_pos):
    # remove the enemy aound if they have only one empty point
    # and current point is the only empty point they have
    # decrese empty point number if it is not the only empty point they have
    if point != None and point.get_color() == enemy_color_value:  
      enemy_group_reporter = self.group_reporters.get(point.get_group_id())
      if enemy_group_reporter != None: 
        # enemy_group_reporter.print_debug_info()
        # print('# empty point of:' + str(point.get_pos()) + " is :" + str(enemy_group_reporter.get_empty_number()))
        if enemy_group_reporter.is_the_only_empty(cur_pos): #get_empty_number() == 1:
          # if the empty point removed is the only empty point enemy group has, start to remove the group
          working_stack = copy.deepcopy(enemy_group_reporter.get_stack(enemy_color_value))
          # print('# trying to remove: ' + str(working_stack))
          for target_pos in working_stack:
            # get target point base on target pos
            target_point = self.all_points.get(target_pos)
            # set target point's color to 0, so it is removed
            target_point.set_color(0)
            # remember last move stones, so that we can caculate ko
            self.last_remove.add(target_pos)
            # set the group id of target point to -1, so that we can upgrade the new empty group
            target_point.set_group_id(-1)
            self.update_removed_empty_point(enemy_color_value, target_point)
        else:
          enemy_group_reporter.remove_empty(cur_color_value, cur_pos)

  def update_removed_empty_point(self, removed_color, target_point):
    # current point was removed from board
    # need to update the black stack and white stack for point around current point.
    up_point = target_point.get_up()
    down_point = target_point.get_down()
    left_point = target_point.get_left()
    right_point = target_point.get_right()

    if up_point != None:
      if up_point.get_group_id() != -1: # not the empty one in the point just removed
        reporter = self.group_reporters.get(up_point.get_group_id())
        reporter.remove_color(removed_color, target_point.get_pos())
    
    if down_point != None:
      if down_point.get_group_id() != -1: # not the empty one in the point just removed
        reporter = self.group_reporters.get(down_point.get_group_id())
        reporter.remove_color(removed_color, target_point.get_pos())

    if left_point != None:
      if left_point.get_group_id() != -1: # not the empty one in the point just removed
        # print('left point is being handled:' + str(left_point.get_pos()))
        reporter = self.group_reporters.get(left_point.get_group_id())
        # print('before remove: ' + str(reporter.get_empty_number()))
        reporter.remove_color(removed_color, target_point.get_pos())
        # print('after remove: ' + str(reporter.get_empty_number()))

    if right_point != None:
      if right_point.get_group_id() != -1: # not the empty one in the point just removed
        reporter = self.group_reporters.get(right_point.get_group_id())
        reporter.remove_color(removed_color, target_point.get_pos())


  def handle_space_point(self, point, working_group_id_set):
    # update the space point around current point
    empty_color = 0
    if point != None:
      if point.get_color() == 0:
        if not point.get_group_id() in working_group_id_set:
          working_group_id = self.get_group_id_index()
          working_group_id_set.add(working_group_id)
          reporter = Reporter(working_group_id, empty_color)
          reporter = self.update_group(point, empty_color, working_group_id, reporter)
          self.group_reporters[working_group_id] = reporter
    return working_group_id_set


  def update_group(self, cur_point, color_value, group_id, reporter):
    
    if cur_point == None:
      # hit the age of board, return directly
      return reporter

    # print ('# update:' + str(cur_point.get_pos()) + ' color:' + str(color_value) + ' group_id:' + str(group_id))
    # add current point into the reporter history base on the value of current color:
    if cur_point.get_color() == 0:
      # print ('# trying to update reporter with color 0')
      # print ('# reporter empty number:' + str(reporter.get_empty_number()))
      
      reporter.add_empty(cur_point.get_pos())
    elif cur_point.get_color() == 1:
      reporter.add_black(cur_point.get_pos())
    elif cur_point.get_color() == 2:
      reporter.add_white(cur_point.get_pos())

    # update the points around current point
    if cur_point.get_color() == color_value and cur_point.get_group_id() != group_id:
      cur_point.set_group_id(group_id)
    
      up_point = cur_point.get_up()
      down_point = cur_point.get_down()
      left_point = cur_point.get_left()
      right_point = cur_point.get_right()

      reporter = self.update_group(up_point, color_value, group_id, reporter)
      reporter = self.update_group(down_point, color_value, group_id, reporter)
      reporter = self.update_group(left_point, color_value, group_id, reporter)
      reporter = self.update_group(right_point, color_value, group_id, reporter)
    
    return reporter

    
  def is_suicide(self, color_value, pos):
    # @todo, implement this method to check whether the move is suicide.
    result = False
    cur_point = self.all_points.get(pos)
    empty_group_reporter = self.group_reporters.get(cur_point.get_group_id())

    if empty_group_reporter.is_the_only_empty(pos):
      if empty_group_reporter.get_stack_len(color_value) == 0:
        # no stone has same color with cur color, it may be suicide
        # need to check whether current move can kill one of the neighbour
        up_point = cur_point.get_up()
        down_point = cur_point.get_down()
        left_point = cur_point.get_left()
        right_point = cur_point.get_right()

        if self.can_kill_neighbour(up_point, color_value, pos) or \
          self.can_kill_neighbour(down_point, color_value, pos) or \
          self.can_kill_neighbour(left_point, color_value, pos) or \
          self.can_kill_neighbour(right_point, color_value, pos):
          result = False
        else:
          result = True
        
      else:
        up_point = cur_point.get_up()
        down_point = cur_point.get_down()
        left_point = cur_point.get_left()
        right_point = cur_point.get_right()

        if self.is_suicide_neighbour(up_point, color_value, pos) and \
           self.is_suicide_neighbour(down_point, color_value, pos) and \
           self.is_suicide_neighbour(left_point, color_value, pos) and \
           self.is_suicide_neighbour(right_point, color_value, pos):
          result = True

    return result

  def is_suicide_neighbour(self, cur_point, color_value, pos):
    result = False
    if cur_point == None:
      result = True
    else:
      if cur_point.get_color() == color_value:
        cur_group_reporter = self.group_reporters.get(cur_point.get_group_id())
        if cur_group_reporter.is_the_only_empty(pos):
          result = True
      else:
        result = True
    return result

  def can_kill_neighbour(self, cur_point, color_value, pos):
    result = False
    if cur_point == None:
      result = False
    else:
      enemy_color_value = self.reverse_color_value(color_value)
      if cur_point.get_color() == enemy_color_value:
        cur_group_reporter = self.group_reporters.get(cur_point.get_group_id())
        if cur_group_reporter.is_the_only_empty(pos):
          result = True
      else:
        result = False
    return result

  def is_ko_by_letter(self, color, pos):
    color_value = self.get_color_value(color)
    return self.is_ko(color_value, pos)

  def is_ko(self, color_value, pos):
    result = False
    if len(self.last_remove) == 1:
      # just one stone was removed in last move, it may be a ko
      if pos in self.last_remove:
        # current position is the only one in last remove, it may be a ko
        (last_move_pos, last_move_color) = self.last_move
        last_move_point = self.all_points.get(last_move_pos)
        group_reporter = self.group_reporters.get(last_move_point.get_group_id())
        enemy_color_value = self.reverse_color_value(color_value)
        if group_reporter.get_stack_len(enemy_color_value) == 1:
          if group_reporter.is_the_only_empty(pos):
            # last move has only one stone in group, and only one empty left, which is current position
            # it is a ko
            result = True
    return result

  def is_move_legal(self, color, pos):
    color_value = self.get_color_value(color)

    # return false if current position is not empty
    cur_point = self.all_points.get(pos)
    if cur_point.get_color() != 0:
      return False

    # return false if current position is a suicide move
    if self.is_suicide(color_value, pos):
      return False

    # return false if current position is a ko
    if self.is_ko(color_value, pos):
      return False

    return True
  
  def is_my_eye(self, color, pos):
    # @todo build the eye checking system 
    return False
    
  def get_group_id_index(self):
    self.group_id_index = self.group_id_index + 1
    return self.group_id_index

  def connect_all_points(self):
    for row in range(self.board_size):
      for col in range(self.board_size):
        cur_point = self.all_points.get((row,col))
        up_point = self.all_points.get((row+1, col))
        down_point = self.all_points.get((row-1, col))
        left_point = self.all_points.get((row, col-1))
        right_point = self.all_points.get((row, col+1))
        cur_point.set_up(up_point)
        cur_point.set_down(down_point)
        cur_point.set_left(left_point)
        cur_point.set_right(right_point)

  def get_color_value(self, color):
    if color == 'b':
      color_value = 1
    elif color == 'w':
      color_value = 2
    else:
      raise ValueError

    return color_value

  def get_enemy_color_value(self, color):
    if color == 'b':
      color_value = 2
    elif color == 'w':
      color_value = 1
    else:
      raise ValueError

    return color_value

  def other_color(self, color):
    '''
    Color of other player
    '''
    if color == 'b':
        return 'w'
    if color == 'w':
        return 'b'

  def reverse_color_value(self, color_value):
    if color_value == 0:
      return 0
    elif color_value == 1:
      return 2
    elif color_value == 2:
      return 1
    else:
      raise ValueError

  def get_score(self):
    valid_group_ids = set()

    for i in range(self.board_size):
      for j in range(self.board_size):
        point = self.all_points.get((i,j))
        cur_group_id = point.get_group_id()
        valid_group_ids.add(cur_group_id)

    total_empty = 0
    total_black = 0
    total_white = 0

    # print ('# group reporter len: ' + str(self.group_reporters))
    for reporter_id in valid_group_ids:
      
      reporter = self.group_reporters.get(reporter_id)
      # print ('# group id: ' + str(reporter_id) + ' group type: ' + str(reporter.get_group_type())),

      (empty_number, black_number, white_number) = reporter.get_score()
      total_empty = total_empty + empty_number
      total_black = total_black + black_number
      total_white = total_white + white_number

      # print ('empty:' + str(empty_number) + '  black:' + str(black_number) + '  white:' + str(white_number))

    total_socre = total_empty + total_black + total_white
    board_score = self.board_size * self.board_size

    if total_socre != board_score:
      print ('# warning: inconsiste status: total/board: ' + str(total_socre) + '/' + str(board_score))

    return (total_empty, total_black, total_white)

  def get(self, pos):
    # for back compatibility
    # return the color letter 'b' or 'w' for current position

    cur_point = self.all_points.get(pos)
    if cur_point.get_color() == 1:
      return 'b'
    elif cur_point.get_color() == 2:
      return 'w'
    else:
      return 'e' # 'e' stand for empty

  def get_liberties(self, pos):
    # for back compatibility
    # return the number of liberties of current position

    cur_point = self.all_points.get(pos)
    cur_group_reporter = self.group_reporters.get(cur_point.get_group_id())
    return cur_group_reporter.get_empty_number()

  def analyst_result(self):
    valid_group_ids = set()

    for i in range(self.board_size):
      for j in range(self.board_size):
        point = self.all_points.get((i,j))
        cur_group_id = point.get_group_id()
        valid_group_ids.add(cur_group_id)

    total_empty = 0
    total_black = 0
    total_white = 0

    # print ('# group reporter len: ' + str(self.group_reporters))
    for reporter_id in valid_group_ids:
      
      reporter = self.group_reporters.get(reporter_id)
      # print ('# group id: ' + str(reporter_id) + ' group type: ' + str(reporter.get_group_type())),

      (empty_number, black_number, white_number) = reporter.get_score()
      if empty_number > 0:
        print('reporter type:' + str(reporter.get_group_type()))
        print('empty:' + str(empty_number))
        print('black:' + str(black_number))
        print('white:' + str(white_number))
        empty_stack = reporter.get_stack(0)
        print('first in empty stack:' + str(empty_stack))

    

  def __str__(self):
    result = '# GoPoints\n'
    display_letter = '.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789'
    display_letter_number = len(display_letter)
    display_letter_index = 0
    group_mapping = {}

    for i in range(self.board_size - 1, -1, -1):
        line = '# '
        for j in range(0, self.board_size):
            this_point = self.all_points.get((i, j))
            if this_point is None:
              line = line + '.'
            else:
              if this_point.get_color() == 1:
                line = line + '*'
              if this_point.get_color() == 2:
                line = line + 'O'
              if this_point.get_color() == 0:
                group_id = this_point.get_group_id()
                group_index = group_mapping.get(group_id)
                if group_index is None:
                  # print('found new group id: ' + str(group_id))
                  group_index = display_letter_index
                  display_letter_index = display_letter_index + 1
                  group_mapping[group_id] = group_index
                cur_letter = display_letter[group_index%display_letter_number]
                line = line + cur_letter
                

        result = result + line + '\n'
    
    (empty_score, black_score, white_score) = self.get_score()
    
    result = result + '# Black:' + str(black_score) + ' White:' + str(white_score) + ' Empty:' + str(empty_score) +'\n'

    return result


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

 
  
class Reporter(object):

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
    
  

