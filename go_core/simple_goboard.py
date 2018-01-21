import numpy as np
import copy
import random


class SimpleGoBoard(object):

  def __init__(self, board_size=19, eye_checking = False):
    # self.board_size = board_size
    self.reset(board_size, eye_checking)

  def reset(self, board_size, eye_checking = False):
    self.board_size = board_size    
    self.eye_checking = eye_checking    
    
    self.board = list()
    self.review_record = list()

    for i in range(self.board_size):
      self.board.append(list())
      self.review_record.append(list())
      for j in range(self.board_size):
        self.board[i].append(0)
        self.review_record[i].append(0)

    self.potential_ko = False
    self.ko_remove = (-1, -1)

    self.empty_space = set()

    for i in range(self.board_size):
      for j in range(self.board_size):
        self.empty_space.add((i,j))
    



  def apply_move(self, color, pos):

    if pos is None:
      # current player pass, just return
      return 

    (row, col) = pos
    color_value = self.get_color_value(color)
    enemy_color_value = self.get_enemy_color_value(color)

    if self.board[row][col] != 0:
      # current point is not empty, return
      return
    
    self.board[row][col] = color_value
    self.empty_space.remove((row, col))
    
    up_removed = self.remove_if_enemy_dead(row+1, col, enemy_color_value)
    right_removed = self.remove_if_enemy_dead(row, col+1, enemy_color_value)
    down_removed = self.remove_if_enemy_dead(row-1, col, enemy_color_value)
    left_removed = self.remove_if_enemy_dead(row, col-1, enemy_color_value)

    if not (up_removed > 0 or right_removed >0 or down_removed > 0 or left_removed > 0):
      suicide = self.remove_if_dead(row, col)
      # dosn't check the suicide move, will just let it die if it is a suicide move
      # if suicide:
      #   # it is suicide
      #   print ('#warning: suicide move detect')
      #   return

    if up_removed+right_removed+down_removed+left_removed > 1:
      # more than one stones were removed, it could not be a potential_ko
      self.potential_ko = False
    else:
      self.potential_ko = True
      if up_removed == 1:
        self.ko_remove = (row+1, col)
      elif right_removed == 1:
        self.ko_remove = (row, col+1)
      elif down_removed == 1:
        self.ko_remove = (row-1, col)
      elif left_removed == 1:
        self.ko_remove = (row, col-1)

    if self.eye_checking:
      # print('# trying to check the eye state')
      self.update_eye_state()

    self.move_history_color_value = color_value
    self.move_history_pos = pos

        
  def remove_if_enemy_dead(self, row, col, enemy_color_value):
    if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
      return 0

    if self.board[row][col] == enemy_color_value:
      return self.remove_if_dead(row, col)
    else:
      return 0
    
  def remove_if_dead(self, row, col):
    # print('# trying to remove if it is dead: ' + str(row) + ',' +str(col))
    if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
      return 0

    target_color_value = self.board[row][col]
    if target_color_value == 0:
      # it is empty, return 0
      # print('# move of: '+ str(cur_move_number) + ' is: ' + str(self.board[cur_move_number][row][col]))
      return 0

    self.reset_review_record()

    # print('# trying to call dead review: ' + str(row) + ',' +str(col))
    is_dead = self.dead_review(row, col, target_color_value)
    # print('# after calling dead review: is dead: ' + str(is_dead))

    if is_dead:
      removed_number = 0
      for i in range(self.board_size):
        for j in range(self.board_size):
          if self.review_record[i][j] == 1:
            self.board[i][j] = 0
            self.empty_space.add((i, j))
            removed_number = removed_number + 1
      return removed_number

    return 0
    
  def dead_review(self, row, col, target_color_value):
    # print('# reviewing: ' + str(row) + ',' + str(col))
    if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
      # current location is out of board, return True
      return True

    if self.review_record[row][col] == 1:
      # this point has been reviewed, return True
      return True

    if self.board[row][col] == target_color_value:
      # color of current stone is target_color_value, need to check the location around.

      # set current location as reviewd
      self.review_record[row][col] = 1

      # start to check the location around
      is_dead = self.dead_review(row+1, col, target_color_value)
      if not is_dead:
        return False

      is_dead = self.dead_review(row, col+1, target_color_value)
      if not is_dead:
        return False
      
      is_dead = self.dead_review(row-1, col, target_color_value)
      if not is_dead:
        return False
      
      is_dead = self.dead_review(row, col-1, target_color_value)
      if not is_dead:
        return False

      # beside the point in review history, all the neighbours are dead, return True
      # print('# stone around are dead')
      return True
    elif self.board[row][col] == 0:
      # current location is empty, target is not dead
      return False
    else:
      # current location has stone with enemy's color, return True
      # print('# current point has enemy stone')
      return True

  # def reset_wall_mark_index(self):
  #   self.wall_mark_index = 0

  # def get_next_wall_mark_index(self):
  #   self.wall_mark_index = self.wall_mark_index + 1
  #   return self.wall_mark_index

   
  def update_eye_state(self, cur_move_number):
    pass

      
  def get_random_empty_space(self):
    return random.sample(self.empty_space, 1)[0]

  def is_my_eye(self, color, pos):
    if not self.eye_checking:
      return False

    if pos is None:
      return False
    color_value = self.get_color_value(color)
    return self.is_simple_true_eye(color_value, pos)

  def is_simple_true_eye(self, color_value, pos):
    if pos is None:
      return False

    (row, col) = pos
    if color_value == 1:
      if self.black_eye_state[self.move_number][row][col] == 1:
        return True
    elif color_value == 2:
      if self.white_eye_state[self.move_number][row][col] == 1:
        return True
    return False
      
  # if current position is empty, not a ko and not suicide, it is legal
  def is_move_legal(self, color, pos):
    
    if pos is None:
      # the position is None, that means this player pass, it is a legal move.
      return True
    
    last_move_number = self.move_number
    cur_move_number = self.move_number + 1

    if cur_move_number >= self.history_length:
      # this board has more than 1024 moves, it is full, just return False
      return False

    (row, col) = pos
    color_value = self.get_color_value(color)
    enemy_color_value = self.get_enemy_color_value(color)

    if self.board[last_move_number][row][col] != 0:
      # current point is not empty, it is illegal, return False
      return False
    
    if self.potential_ko:
      # if last move is a potential ko move
      if color_value != self.move_history_color_value[last_move_number]:
        # last move is played by enemy
        if self.ko_remove == pos:
          # it is a Ko, it is illegal, return False
          return False

    # copy last state to current state, for suicide checking
    self.board[cur_move_number] = self.board[last_move_number]

    self.board[cur_move_number][row][col] = color_value

    up_removed = self.remove_if_dead(cur_move_number, row+1, col)
    right_removed = self.remove_if_dead(cur_move_number, row, col+1)
    down_removed = self.remove_if_dead(cur_move_number, row-1, col)
    left_removed = self.remove_if_dead(cur_move_number, row, col-1)

    if not (up_removed > 0 or right_removed >0 or down_removed > 0 or left_removed > 0):
      suicide_number = self.remove_if_dead(cur_move_number, row, col)
      if suicide_number > 0:
        # it is suicide, return False
        return False

    #make sure that we didn't update the self.move_number here, as it is a simulation for legal checking

    # it is not empty, it is not a ko, and it it not suicide either, it is legal
    return True


  def reset_review_record(self):
    for i in range(self.board_size):
      for j in range(self.board_size):
        self.review_record[i][j] = 0


      
  def get_cur_move_color(self):
    if self.move_number == 0:
      return 'b'
    else:
      if self.move_history_color_value == 1:
        return 'w'
      elif self.move_history_color_value == 2:
        return 'b'
      else:
        return 'e' # error

  # convert the color letter to color value
  def get_color_value(self, color):
    if color == 'b':
      color_value = 1
    elif color == 'w':
      color_value = 2
    else:
      raise ValueError

    return color_value

  # convert the color letter to enemy's color value
  def get_enemy_color_value(self, color):
    if color == 'b':
      color_value = 2
    elif color == 'w':
      color_value = 1
    else:
      raise ValueError
    return color_value

  # reverse the color letter
  def other_color(self, color):
    '''
    Color of other player
    '''
    if color == 'b':
        return 'w'
    if color == 'w':
        return 'b'

  # reverse the color_value 
  def reverse_color_value(self, color_value):
    if color_value == 0:
      return 0
    elif color_value == 1:
      return 2
    elif color_value == 2:
      return 1
    else:
      raise ValueError

  # get score of current board
  # return: (total_empty, total_black, total_white)
  def get_score(self):
    # @todo, we didn't implement get_score here
    total_empty = 0
    total_black = 0
    total_white = 0

    return (total_empty, total_black, total_white)

  # back compatiblility code: get letter of current position
  ##################################
  def get(self, pos):
    # for back compatibility
    # return the color letter 'b' or 'w' for current position

    (row, col) = pos

    if self.board[self.move_number][row][col] == 1:
      return 'b'
    elif self.board[self.move_number][row][col] == 2:
      return 'w'
    else:
      return 'e' # 'e' stand for empty

  def get_array_result(self):
    result = self.board[self.move_number]
    return result

  def get_legal_empty_space(self, color):
    return self.get_legal_empty_space_at(self.move_number, color)

  def get_legal_empty_space_at(self, cur_move_number, color):
    result = list()
    if cur_move_number >= self.history_length:
      return result
    for i in range(self.board_size):
      for j in range(self.board_size):
        if self.board[cur_move_number][i][j] == 0:
          # if self.is_move_legal(color, (i, j)):
            result.append((i, j))
    return result

  def get_random_next(self, color):
    empty_locations = self.get_legal_empty_space(color)

    if len(empty_locations) < 1:
      return None

    empty_number = len(empty_locations)

    random_number = random.randint(0, empty_number-1)

    return empty_locations[random_number]


  def get_move_array(self, history_length, color):
    result = np.zeros((history_length*2+1, self.board_size, self.board_size), dtype=int)
    color_value = self.get_color_value(color)

    for i in range(history_length):
      cur_point = self.move_number - i
      if cur_point < 0:
        result[i*2] = self.zero_array
        result[i*2 + 1] = self.zero_array
      else:
        if color_value == 1:
          result[i*2] = self.black_history[cur_point]
          result[i*2 + 1] = self.white_history[cur_point]
        elif color_value == 2:
          result[i*2] = self.white_history[cur_point]
          result[i*2 + 1] = self.black_history[cur_point]

    if color_value == 1:
      result[-1] = self.one_array
    elif color_value == 2:
      result[-1] = self.zero_array

    return result

  def start_simulating(self):
    self.is_simulating = True
    self.real_move_number = self.move_number

  def stop_simulating(self):
    self.move_number = self.real_move_number
    self.is_simulating = False


  # debuging function: return string representing current board
  ##############################
  def __str__(self):
    result = ''
    result = result + self.get_standard_debug_string()
    # result = result + self.get_wall_mark_debug_string()
    # result = result + self.get_eye_debug_string()
    # result = result + self.get_history_debug_string()
    
    return result

  def get_standard_debug_string(self):
    result = '# GoPoints\n'

  
    for i in range(self.board_size - 1, -1, -1):
        line = '# '
        for j in range(0, self.board_size):
              if self.board[i][j] == 1:
                line = line + '*'
              if self.board[i][j] == 2:
                line = line + 'O'
              if self.board[i][j] == 0:
                line = line + '.'

        result = result + line + '\n'
    
    return result

  def get_eye_debug_string(self):
    result = '# ArrayGoBoard, eye:\n'
    result = result + '# Black eye:\n'
    for i in range(self.board_size - 1, -1, -1):
        line = '# '
        for j in range(0, self.board_size):
              line = line + str(self.black_eye_state[self.move_number][i][j])

        result = result + line + '\n'

    result = result + '# White eye:\n'
    for i in range(self.board_size - 1, -1, -1):
        line = '# '
        for j in range(0, self.board_size):
              line = line + str(self.white_eye_state[self.move_number][i][j])

        result = result + line + '\n'
    
    return result

  def get_history_debug_string(self):
    result = '# ArrayGoBoard, history\n'
    
    result = result + '# Black history:\n'
    for i in range(self.board_size - 1, -1, -1):
        line = '# '
        for j in range(0, self.board_size):
              line = line + str(self.black_history[self.move_number][i][j])

        result = result + line + '\n'

    result = result + '# White history:\n'
    for i in range(self.board_size - 1, -1, -1):
        line = '# '
        for j in range(0, self.board_size):
              line = line + str(self.white_history[self.move_number][i][j])

        result = result + line + '\n'
    
    return result

  def get_wall_mark_debug_string(self):
    result = '# ArrayGoBoard, wall mark:\n'
    
    if self.move_number < 1:
      result = result + 'No wall mark records now.'
    else:
      for i in range(self.board_size - 1, -1, -1):
          line = '# '
          for j in range(0, self.board_size):
                line = line + str(self.wall_mark[self.move_number][i][j])

          result = result + line + '\n'

    
    
    return result




