import numpy as np
import copy


class ArrayGoBoard(object):

  def __init__(self, board_size=19):
    self.board_size = board_size
    self.reset(board_size)

  def reset(self, board_size):
    self.board_size = board_size        
    self.history_length = 1024
    self.move_number = 0
    self.board = np.zeros((self.history_length, self.board_size, self.board_size), dtype=int)
    self.review_record = np.zeros((self.board_size, self.board_size), dtype=int)
    self.potential_ko = False
    self.ko_remove = (-1, -1)
    self.last_move = (-1, -1)
    self.last_move_color_value = 0

    



  def apply_move(self, color, pos):
    # apply move to position
    # print('# applying move: ' + color + "  " + str(pos))
    last_move_number = self.move_number
    cur_move_number = self.move_number + 1

    if cur_move_number >= self.history_length:
      # this board has more than 1024 moves, it is full, just return
      print('#warning: array go board is full')
      return

    (row, col) = pos
    color_value = self.get_color_value(color)
    enemy_color_value = self.get_enemy_color_value(color)

    if self.board[last_move_number][row][col] != 0:
      # current point is not empty, return
      return
    
    # copy last state to current state, get ready to apply move
    self.board[cur_move_number] = self.board[last_move_number]

    # for performance, doesn't check ko here, will apply it anyway.
    # need to make sure it is not a ko before we call apply_move
    # if self.is_ko(color, pos):
    #   return

    self.board[cur_move_number][row][col] = color_value

    
    up_removed = self.remove_if_enemy_dead(cur_move_number, row+1, col, enemy_color_value)
    right_removed = self.remove_if_enemy_dead(cur_move_number, row, col+1, enemy_color_value)
    down_removed = self.remove_if_enemy_dead(cur_move_number, row-1, col, enemy_color_value)
    left_removed = self.remove_if_enemy_dead(cur_move_number, row, col-1, enemy_color_value)

    if not (up_removed > 0 or right_removed >0 or down_removed > 0 or left_removed > 0):
      suicide = self.remove_if_dead(cur_move_number, row, col)
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

    self.move_number = self.move_number + 1
    self.last_move_color_value = color_value
    self.last_move_pos = pos
        
  def remove_if_enemy_dead(self, cur_move_number, row, col, enemy_color_value):
    if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
      return 0

    if self.board[cur_move_number][row][col] == enemy_color_value:
      return self.remove_if_dead(cur_move_number, row, col)
    else:
      return 0
    
  def remove_if_dead(self, cur_move_number, row, col):
    # print('# trying to remove if it is dead: ' + str(row) + ',' +str(col))
    if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
      return 0

    target_color_value = self.board[cur_move_number][row][col]
    if target_color_value == 0:
      # it is empty, return 0
      # print('# move of: '+ str(cur_move_number) + ' is: ' + str(self.board[cur_move_number][row][col]))
      return 0

    self.review_record = np.zeros((self.board_size, self.board_size), dtype=int)

    # print('# trying to call dead review: ' + str(row) + ',' +str(col))
    is_dead = self.dead_review(cur_move_number, row, col, target_color_value)
    # print('# after calling dead review: is dead: ' + str(is_dead))

    if is_dead:
      removed_number = 0
      for i in range(self.board_size):
        for j in range(self.board_size):
          if self.review_record[i][j] == 1:
            self.board[cur_move_number][i][j] = 0
            removed_number = removed_number + 1
      return removed_number

    return 0
    
  def dead_review(self, cur_move_number, row, col, target_color_value):
    # print('# reviewing: ' + str(row) + ',' + str(col))
    if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
      # current location is out of board, return True
      return True

    if self.review_record[row][col] == 1:
      # this point has been reviewed, return True
      return True

    if self.board[cur_move_number][row][col] == target_color_value:
      # color of current stone is target_color_value, need to check the location around.

      # set current location as reviewd
      self.review_record[row][col] = 1

      # start to check the location around
      is_dead = self.dead_review(cur_move_number, row+1, col, target_color_value)
      if not is_dead:
        return False

      is_dead = self.dead_review(cur_move_number, row, col+1, target_color_value)
      if not is_dead:
        return False
      
      is_dead = self.dead_review(cur_move_number, row-1, col, target_color_value)
      if not is_dead:
        return False
      
      is_dead = self.dead_review(cur_move_number, row, col-1, target_color_value)
      if not is_dead:
        return False

      # beside the point in review history, all the neighbours are dead, return True
      # print('# stone around are dead')
      return True
    elif self.board[cur_move_number][row][col] == 0:
      # current location is empty, target is not dead
      return False
    else:
      # current location has stone with enemy's color, return True
      # print('# current point has enemy stone')
      return True



   
  

  def is_true_eye(self, color, pos):
    (row, col) = pos
    if color == 1:
      if self.black_eyes[row][col] > 0:
        return True
    elif color == 2:
      if self.white_eyes[row][col] > 0:
        return True
    return False

  def is_simple_true_eye(self, color, pos):
    (row, col) = pos
    if color == 1:
      if self.black_eyes[row][col] == 1:
        return True
    elif color == 2:
      if self.white_eyes[row][col] == 1:
        return True
    return False
      
  # if current position is empty, not a ko and not suicide, it is legal
  def is_move_legal(self, color, pos):
    # apply move to position
    # print('# applying move: ' + color + "  " + str(pos))
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
      if color_value != self.last_move_color_value:
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

  def update_eye_state(self):
    self.black_eye_state = np.zeros((self.board_size, self.board_size), dtype=int)
    self.black_eye_state = np.zeros((self.board_size, self.board_size), dtype=int)
    
    self.potential_black_eyes = set()
    self.potential_white_eyes = set()

    for i in range(self.board_size):
      for j in range(self.board_size):
        if self.is_eye(i, j, 1):
          eye_type = self.get_eye_type(i, j, 1)
          if eye_type == 1:
            self.black_eye_state[i][j] = 1
          elif eye_type == 2:
            self.black_eye_state[i][j] = 2
            self.potential_black_eyes.add((i,j))
    
    for potential_location in self.potential_black_eyes:
      self.check_potential_location(check_potential_location, 1)

    for i in range(self.board_size):
      for j in range(self.board_size):
        if self.is_eye(i, j, 2):
          eye_type = self.get_eye_type(i, j, 2)
          if eye_type == 1:
            self.white_eye_state[i][j] = 1
          elif eye_type == 2:
            self.white_eye_state[i][j] = 2
            self.potential_white_eyes.add((i,j))
    
    for potential_location in self.potential_white_eyes:
      self.check_potential_location(check_potential_location, 2)
            
          
  def is_eye(row, col, color_value):
    if self.board[self.move_number][row][col] != 0:
      return False

    if row-1 >= 0:
      if self.board[self.move_number][row-1][col] != color_value:
        return False

    if col-1 >= 0:
      if self.board[self.move_number][row][col-1] != color_value:
        return False

    if row+1 < self.board_size:
      if self.board[self.move_number][row+1][col] != color_value:
        return False

    if col+1 < self.board_size:
      if self.board[self.move_number][row][col+1] != color_value:
        return False

    return True
  
  def get_eye_type(row, col, color_value):
    # detect the type of eye: 0: false eye, 1: true eye, 2: potential true eye

    enemy_color_value = self.reverse_color_value(color_value)

    if row-1 < 0:
      if col-1 < 0:
        if self.board[self.move_number][row+1][col+1] == color_value:
          return 1
        elif self.board[self.move_number][row+1][col+1] == 0:
          return 2
        else:
          return 0
      elif col+1 >=self.board_size:
        if self.board[self.move_number][row+1][col-1] == color_value:
          return 1
        elif self.board[self.move_number][row+1][col-1] == 0:
          return 2
        else:
          return 0
      else:
        if self.board[self.move_number][row+1][col-1] == color_value and \
           self.board[self.move_number][row+1][col+1] == color_value:
          return 1
        elif self.board[self.move_number][row+1][col-1] == enemy_color_value or \
           self.board[self.move_number][row+1][col+1] == enemy_color_value:
          return 0
        else:
          return 2
    elif row+1 >= self.board_size:
      if col-1 < 0:
        if self.board[self.move_number][row-1][col+1] == color_value:
          return 1
        elif self.board[self.move_number][row-1][col+1] == 0:
          return 2
        else:
          return 0
      elif col+1 >= self.board_size:
        if self.board[self.move_number][row-1][col-1] == color_value:
          return 1
        elif self.board[self.move_number][row-1][col-1] == 0:
          return 2
        else:
          return 0
      else:
        if self.board[self.move_number][row-1][col-1] == color_value and \
           self.board[self.move_number][row-1][col+1] == color_value:
          return 1
        elif self.board[self.move_number][row-1][col-1] == enemy_color_value or \
           self.board[self.move_number][row-1][col+1] == enemy_color_value:
          return 0
        else:
          return 2
    else:
      if col-1 < 0:
        if self.board[self.move_number][row-1][col+1] == color_value and \
           self.board[self.move_number][row+1][col+1] == color_value:
          return 1
        elif self.board[self.move_number][row-1][col+1] == enemy_color_value or \
           self.board[self.move_number][row+1][col+1] == enemy_color_value:
          return 0
        else:
          return 2
      elif col+1 >= self.board_size:
        if self.board[self.move_number][row-1][col-1] == color_value and \
           self.board[self.move_number][row+1][col-1] == color_value:
          return 1
        elif self.board[self.move_number][row-1][col-1] == enemy_color_value or \
           self.board[self.move_number][row+1][col-1] == enemy_color_value:
          return 0
        else:
          return 2
      else:
        number_list = [-1, 1]

        same_color_number = 0
        enemy_color_number = 0
        empty_color_number = 0
        
        for i in number_list:
          for j in number_list:
            if self.board[self.move_number][i][j] == color_value:
              same_color_number = same_color_number + 1
            elif self.board[self.move_number][i][j] == enemy_color_value:
              enemy_color_number = enemy_color_value + 1
            else:
              empty_color_number = empty_color_number + 1

        if same_color_number >= 3:
          return 1
        else:
          if enemy_color_number >= 2:
            return 0
          else:
            return 2

    # the following code shouldn't be reach, 
    # return 0 to make sure there is no type error when anything wired happened
    return 0
  
  def check_potential_location(self, pos, target_color_value):
    self.reset_review_record()
    target_state = None
    if target_color_value == 1: 
      target_state = self.black_eye_state
    elif target_state == 2:
      target_state = self.white_eye_state
    else:
      return
    
    (row, col) = pos
    if target_state[row][col] != 2:
      # current location is not a potential eye now, just return
      return

    enemy_color_value = self.reverse_color_value(target_color_value)
    is_true_eye = self.eye_review(row, col, target_color_value, enemy_color_value, target_state)

    
    for i in range(self.board_size):
      for j in range(self.board_size):
        if self.review_record[i][j] == 1:
          if is_true_eye:
            target_state[i][j] = 1
          else:
            target_state[i][j] = 0
          
      


  def eye_review(self, row, col, target_color_value, enemy_color_value, target_state):

    
    if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
      # current location is out of board, return True
      return True

    if self.review_record[row][col] == 1:
      return True

    

    if target_state[row][col] == 1:
      #current location is a true eye, return True
      return True
    elif target_state[row][col] == 2:
      # current location is a new potential true eye, check the location around

      self.review_record[row][col] = 1

      bad_shoulder_number = 0
      is_true_eye = self.eye_review(row-1, col-1, target_color_value, enemy_color_value, target_state)
      if not is_true_eye:
        bad_shoulder_number = bad_shoulder_number + 1

      is_true_eye = self.eye_review(row-1, col+1, target_color_value, enemy_color_value, target_state)
      if not is_true_eye:
        bad_shoulder_number = bad_shoulder_number + 1

      is_true_eye = self.eye_review(row+1, col-1, target_color_value, enemy_color_value, target_state)
      if not is_true_eye:
        bad_shoulder_number = bad_shoulder_number + 1

      is_true_eye = self.eye_review(row+1, col+1, target_color_value, enemy_color_value, target_state)
      if not is_true_eye:
        bad_shoulder_number = bad_shoulder_number + 1

      if bad_shoulder_number > 1:
        return False
      else:
        return True
    else:
      # current location is not an eye, check the location state
      # if current location is empty, it is a bad_shoulder, return false
      # if current location has a stone with enemy's color, it is a bad_shoulder, return false
      # if current location has a stone with current color, it is a good_shoulder, return true
      if self.board[self.move_number][row][col] == 0:
        return False
      elif self.board[self.move_number][row][col] == enemy_color_value:
        return False
      elif self.board[self.move_number][row][col] == target_color_value:
        return True





  def reset_review_record(self):
    self.review_record = np.zeros((self.board_size, self.board_size), dtype=int)

  def is_my_simple_eye(self, color, pos):
    
    color_value = self.get_color_value(color)
    (row, col) = pos

    if color_value == 1:
      if self.black_eye_state[row][col] == 1:
        return True
      else:
        return False
    elif color_value == 2:
      if self.white_eye_state[row][col] == 1:
        return True
      else:
        return False

      
  

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

  # debuging function: return string representing current board
  ##############################
  def __str__(self):
    return self.get_standard_debug_string()

  def get_standard_debug_string(self):
    result = '# GoPoints\n'
    
    for i in range(self.board_size - 1, -1, -1):
        line = '# '
        for j in range(0, self.board_size):
              if self.board[self.move_number][i][j] == 1:
                line = line + '*'
              if self.board[self.move_number][i][j] == 2:
                line = line + 'O'
              if self.board[self.move_number][i][j] == 0:
                line = line + '.'

        result = result + line + '\n'
    
    return result

 



