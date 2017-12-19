import numpy as np
import copy

from go_core.group import Group
from go_core.point import Point

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

    self.group_records = {}

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
    group_record = Group(working_group_id, empty_color)
    group_record = self.update_group(point, empty_color, working_group_id, group_record)
    # print ('# update finished, group_record empty number: ' + str(group_record.get_empty_number()))
    self.group_records[working_group_id] = group_record

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
    group_record = Group(point_group_id, cur_color_value)
    # add current point into the new group at first
    # group_record.add_point(cur_color_value, pos)

    # try to merge the friend point around
    # group_record = self.handle_friend_point(up_point, cur_color_value, point_group_id, group_record)
    # group_record = self.handle_friend_point(down_point, cur_color_value, point_group_id, group_record)
    # group_record = self.handle_friend_point(left_point, cur_color_value, point_group_id, group_record)
    # group_record = self.handle_friend_point(right_point, cur_color_value, point_group_id, group_record)
    # save the group_record of current point

    group_record = self.update_group(cur_point, cur_color_value, point_group_id, group_record)
    self.group_records[point_group_id] = group_record
    # group_record.print_debug_info()
    
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

  def handle_friend_point(self, point, cur_color_value, point_group_id, group_record):
    if point != None and point.get_color() == cur_color_value:  
        group_record = self.update_group(point, cur_color_value, point_group_id, group_record)
    return group_record

  def handle_enemy_point(self, point, enemy_color_value, cur_color_value, cur_pos):
    # remove the enemy aound if they have only one empty point
    # and current point is the only empty point they have
    # decrese empty point number if it is not the only empty point they have
    if point != None and point.get_color() == enemy_color_value:  
      enemy_group_record = self.group_records.get(point.get_group_id())
      if enemy_group_record != None: 
        # enemy_group_record.print_debug_info()
        # print('# empty point of:' + str(point.get_pos()) + " is :" + str(enemy_group_record.get_empty_number()))
        if enemy_group_record.is_the_only_empty(cur_pos): #get_empty_number() == 1:
          # if the empty point removed is the only empty point enemy group has, start to remove the group
          working_stack = copy.deepcopy(enemy_group_record.get_stack(enemy_color_value))
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
          enemy_group_record.remove_empty(cur_color_value, cur_pos)

  def update_removed_empty_point(self, removed_color, target_point):
    # current point was removed from board
    # need to update the black stack and white stack for point around current point.
    up_point = target_point.get_up()
    down_point = target_point.get_down()
    left_point = target_point.get_left()
    right_point = target_point.get_right()

    if up_point != None:
      if up_point.get_group_id() != -1: # not the empty one in the point just removed
        group_record = self.group_records.get(up_point.get_group_id())
        group_record.remove_color(removed_color, target_point.get_pos())
    
    if down_point != None:
      if down_point.get_group_id() != -1: # not the empty one in the point just removed
        group_record = self.group_records.get(down_point.get_group_id())
        group_record.remove_color(removed_color, target_point.get_pos())

    if left_point != None:
      if left_point.get_group_id() != -1: # not the empty one in the point just removed
        # print('left point is being handled:' + str(left_point.get_pos()))
        group_record = self.group_records.get(left_point.get_group_id())
        # print('before remove: ' + str(group_record.get_empty_number()))
        group_record.remove_color(removed_color, target_point.get_pos())
        # print('after remove: ' + str(group_record.get_empty_number()))

    if right_point != None:
      if right_point.get_group_id() != -1: # not the empty one in the point just removed
        group_record = self.group_records.get(right_point.get_group_id())
        group_record.remove_color(removed_color, target_point.get_pos())


  def handle_space_point(self, point, working_group_id_set):
    # update the space point around current point
    empty_color = 0
    if point != None:
      if point.get_color() == 0:
        if not point.get_group_id() in working_group_id_set:
          working_group_id = self.get_group_id_index()
          working_group_id_set.add(working_group_id)
          group_record = Group(working_group_id, empty_color)
          group_record = self.update_group(point, empty_color, working_group_id, group_record)
          self.group_records[working_group_id] = group_record
    return working_group_id_set


  def update_group(self, cur_point, color_value, group_id, group_record):
    
    if cur_point == None:
      # hit the age of board, return directly
      return group_record

    # print ('# update:' + str(cur_point.get_pos()) + ' color:' + str(color_value) + ' group_id:' + str(group_id))
    # add current point into the group_record history base on the value of current color:
    if cur_point.get_color() == 0:
      # print ('# trying to update group_record with color 0')
      # print ('# group_record empty number:' + str(group_record.get_empty_number()))
      
      group_record.add_empty(cur_point.get_pos())
    elif cur_point.get_color() == 1:
      group_record.add_black(cur_point.get_pos())
    elif cur_point.get_color() == 2:
      group_record.add_white(cur_point.get_pos())

    # update the points around current point
    if cur_point.get_color() == color_value and cur_point.get_group_id() != group_id:
      cur_point.set_group_id(group_id)
    
      up_point = cur_point.get_up()
      down_point = cur_point.get_down()
      left_point = cur_point.get_left()
      right_point = cur_point.get_right()

      group_record = self.update_group(up_point, color_value, group_id, group_record)
      group_record = self.update_group(down_point, color_value, group_id, group_record)
      group_record = self.update_group(left_point, color_value, group_id, group_record)
      group_record = self.update_group(right_point, color_value, group_id, group_record)
    
    return group_record

    
  def is_suicide(self, color_value, pos):
    # @todo, implement this method to check whether the move is suicide.
    result = False
    cur_point = self.all_points.get(pos)
    empty_group_record = self.group_records.get(cur_point.get_group_id())

    if empty_group_record.is_the_only_empty(pos):
      if empty_group_record.get_stack_len(color_value) == 0:
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
        cur_group_record = self.group_records.get(cur_point.get_group_id())
        if cur_group_record.is_the_only_empty(pos):
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
        cur_group_record = self.group_records.get(cur_point.get_group_id())
        if cur_group_record.is_the_only_empty(pos):
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
        group_record = self.group_records.get(last_move_point.get_group_id())
        enemy_color_value = self.reverse_color_value(color_value)
        if group_record.get_stack_len(enemy_color_value) == 1:
          if group_record.is_the_only_empty(pos):
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

    # print ('# group group_record len: ' + str(self.group_records))
    for group_record_id in valid_group_ids:
      
      group_record = self.group_records.get(group_record_id)
      # print ('# group id: ' + str(group_record_id) + ' group type: ' + str(group_record.get_group_type())),

      (empty_number, black_number, white_number) = group_record.get_score()
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
    cur_group_record = self.group_records.get(cur_point.get_group_id())
    return cur_group_record.get_empty_number()

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

    # print ('# group group_record len: ' + str(self.group_records))
    for group_record_id in valid_group_ids:
      
      group_record = self.group_records.get(group_record_id)
      # print ('# group id: ' + str(group_record_id) + ' group type: ' + str(group_record.get_group_type())),

      (empty_number, black_number, white_number) = group_record.get_score()
      if empty_number > 0:
        print('group_record type:' + str(group_record.get_group_type()))
        print('empty:' + str(empty_number))
        print('black:' + str(black_number))
        print('white:' + str(white_number))
        empty_stack = group_record.get_stack(0)
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






