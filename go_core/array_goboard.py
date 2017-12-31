import numpy as np
import copy


class ArrayGoBoard(object):

  def __init__(self, board_size=19, eye_checking = False):
    self.board_size = board_size
    self.reset(board_size, eye_checking)

  def reset(self, board_size, eye_checking = False):
    self.board_size = board_size        
    self.history_length = 1024
    self.move_number = 0
    self.is_simulating = False
    self.simulate_move_number = 0

    self.board = np.zeros((self.history_length, self.board_size, self.board_size), dtype=int)
    self.black_history = np.zeros((self.history_length, self.board_size, self.board_size), dtype=int)
    self.white_history = np.zeros((self.history_length, self.board_size, self.board_size), dtype=int)
    self.potential_ko = list()
    self.ko_remove = list()
    self.move_history_pos = list()
    self.move_history_color_value = list()
    for i in range(self.history_length):
      self.potential_ko.append(False)
      self.ko_remove.append((-1, -1))
      self.move_history_pos.append((-1, -1))
      self.move_history_color_value.append(0)

    self.black_eye_state = np.zeros((self.history_length, self.board_size, self.board_size), dtype=int)
    self.white_eye_state = np.zeros((self.history_length, self.board_size, self.board_size), dtype=int)
    
    self.wall_mark = np.zeros((self.history_length, self.board_size, self.board_size), dtype=int)
    self.wall_mark_index = 0

    self.review_record = np.zeros((self.board_size, self.board_size), dtype=int)
    
    self.eye_checking = eye_checking

    self.zero_array = np.zeros((self.board_size, self.board_size), dtype=int)
    self.one_array = np.ones((self.board_size, self.board_size), dtype=int)
    
    



  def apply_move(self, color, pos):

    if pos is None:
      # current player pass, just return
      return 

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
    self.black_history[cur_move_number] = self.black_history[last_move_number]
    self.white_history[cur_move_number] = self.white_history[last_move_number]

    # for performance, doesn't check ko here, will apply it anyway.
    # need to make sure it is not a ko before we call apply_move
    # if self.is_ko(color, pos):
    #   return

    self.board[cur_move_number][row][col] = color_value
    if color_value == 1:
      self.black_history[cur_move_number][row][col] = 1
    elif color_value == 2:
      self.white_history[cur_move_number][row][col] = 1
    
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
      self.potential_ko[cur_move_number] = False
    else:
      self.potential_ko[cur_move_number] = True
      if up_removed == 1:
        self.ko_remove[cur_move_number] = (row+1, col)
      elif right_removed == 1:
        self.ko_remove[cur_move_number] = (row, col+1)
      elif down_removed == 1:
        self.ko_remove[cur_move_number] = (row-1, col)
      elif left_removed == 1:
        self.ko_remove[cur_move_number] = (row, col-1)

    if self.eye_checking:
      # print('# trying to check the eye state')
      self.update_eye_state(cur_move_number)

    self.move_history_color_value[cur_move_number] = color_value
    self.move_history_pos[cur_move_number] = pos

    self.move_number = self.move_number + 1
    
        
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
            if target_color_value == 1:
              self.black_history[cur_move_number][i][j] = 0
            elif target_color_value == 2:
              self.white_history[cur_move_number][i][j] = 0
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

  def reset_wall_mark_index(self):
    self.wall_mark_index = 0

  def get_next_wall_mark_index(self):
    self.wall_mark_index = self.wall_mark_index + 1
    return self.wall_mark_index

   
  def update_eye_state(self, cur_move_number):
    self.black_eye_state[cur_move_number] = np.zeros((self.board_size, self.board_size), dtype=int)
    self.white_eye_state[cur_move_number] = np.zeros((self.board_size, self.board_size), dtype=int)

    self.wall_mark[cur_move_number] = np.zeros((self.board_size, self.board_size), dtype=int)
    self.reset_wall_mark_index()
    
    self.potential_black_eyes = set()
    self.potential_white_eyes = set()

    permanent_black_wall = set()
    permanent_white_wall = set()

    black_eye_with_wall = {}
    white_eye_with_wall = {}

    black_wall_with_eye = {}
    white_wall_with_eye = {}

    for i in range(self.board_size):
      for j in range(self.board_size):
        if self.is_eye(cur_move_number, i, j, 1):
          self.wall_review(cur_move_number, i, j, 1)
          wall_list = self.get_wall(cur_move_number, i, j, 1)
          if len(wall_list) == 0:
            print('# warning, detect an eye with no wall!')
          elif len(wall_list) == 1:
            self.black_eye_state[cur_move_number][i][j] = 1
            permanent_black_wall.add(wall_list[0])
          elif len(wall_list) > 1:
            black_eye_with_wall[(i, j)] = wall_list
            for wall_id in wall_list:
              eye_list = black_wall_with_eye.get(wall_id)
              if eye_list is None:
                new_eye_list = set()
                new_eye_list.add((i,j))
                black_wall_with_eye[wall_id] = new_eye_list
                # print('# after adding first eye:' + str(black_wall_with_eye[wall_id]))
              else:
                eye_list.add((i,j))
                black_wall_with_eye[wall_id] = eye_list
                # print('# after adding new eye:' + str(black_wall_with_eye[wall_id]))
        elif self.is_eye(cur_move_number, i, j, 2): 
          self.wall_review(cur_move_number, i, j, 2)
          wall_list = self.get_wall(cur_move_number, i, j, 2)
          if len(wall_list) == 0:
            print('# warning, detect an eye with no wall!')
          elif len(wall_list) == 1:
            self.white_eye_state[cur_move_number][i][j] = 1
            permanent_white_wall.add(wall_list[0])
          elif len(wall_list) > 1:
            white_eye_with_wall[(i, j)] = wall_list
            for wall_id in wall_list:
              eye_list = white_wall_with_eye.get(wall_id)
              if eye_list is None:
                new_eye_list = set()
                new_eye_list.add((i,j))
                white_wall_with_eye[wall_id] = new_eye_list
                # print('# after adding first eye:' + str(white_wall_with_eye[wall_id]))
              else:
                eye_list.add((i,j))
                white_wall_with_eye[wall_id] = eye_list
                # print('# after adding new eye:' + str(white_wall_with_eye[wall_id]))

    has_dead_wall = True
    while has_dead_wall:
      has_dead_wall = False
      # print('# black wall with eye: ' + str(black_wall_with_eye))
      # print('# black eye with wall: ' + str(black_eye_with_wall))
      
      wall_ids = black_wall_with_eye.keys()
      for wall_id in wall_ids:
        if not wall_id in permanent_black_wall:
          cur_eyes = black_wall_with_eye.get(wall_id)
          if not cur_eyes is None:
            if len(cur_eyes) == 1:
              target_eye = list(cur_eyes)[0]
              remote_walls = black_eye_with_wall.get(target_eye)
              if not remote_walls is None:
                for wall_id in remote_walls:
                  if wall_id in black_wall_with_eye.keys():
                    del black_wall_with_eye[wall_id]
              
              if target_eye in black_eye_with_wall.keys():
                del black_eye_with_wall[target_eye]

              has_dead_wall = True

    for true_eye in black_eye_with_wall.keys():
      (row, col) = true_eye
      self.black_eye_state[cur_move_number][row][col] = 1

    has_dead_wall = True
    while has_dead_wall:
      has_dead_wall = False
      # print('# black wall with eye: ' + str(black_wall_with_eye))
      # print('# black eye with wall: ' + str(black_eye_with_wall))
      
      wall_ids = white_wall_with_eye.keys()
      for wall_id in wall_ids:
        if not wall_id in permanent_white_wall:
          cur_eyes = white_wall_with_eye.get(wall_id)
          if not cur_eyes is None:
            if len(cur_eyes) == 1:
              target_eye = list(cur_eyes)[0]
              remote_walls = white_eye_with_wall.get(target_eye)
              if not remote_walls is None:
                for wall_id in remote_walls:
                  if wall_id in white_wall_with_eye.keys():
                    del white_wall_with_eye[wall_id]
              
              if target_eye in white_eye_with_wall.keys():
                del white_eye_with_wall[target_eye]

              has_dead_wall = True

    for true_eye in white_eye_with_wall.keys():
      (row, col) = true_eye
      self.white_eye_state[cur_move_number][row][col] = 1

      


    #       eye_type = self.get_eye_type(cur_move_number, i, j, 1)
    #       if eye_type == 1:
    #         self.black_eye_state[cur_move_number][i][j] = 1
    #       elif eye_type == 2:
    #         self.black_eye_state[cur_move_number][i][j] = 2
    #         self.potential_black_eyes.add((i,j))
    
    # for potential_location in self.potential_black_eyes:
    #   self.check_potential_location(cur_move_number, potential_location, 1)

    # for i in range(self.board_size):
    #   for j in range(self.board_size):
    #     if self.is_eye(cur_move_number, i, j, 2):
    #       eye_type = self.get_eye_type(cur_move_number, i, j, 2)
    #       if eye_type == 1:
    #         self.white_eye_state[cur_move_number][i][j] = 1
    #       elif eye_type == 2:
    #         self.white_eye_state[cur_move_number][i][j] = 2
    #         self.potential_white_eyes.add((i,j))
    
    # for potential_location in self.potential_white_eyes:
    #   self.check_potential_location(cur_move_number, potential_location, 2)
            
          
  def is_eye(self, cur_move_number, row, col, color_value):
    # print('# trying to get eye type:' + str(cur_move_number) + ' ' + str((row, col, color_value)))
    if self.board[cur_move_number][row][col] != 0:
      return False

    if row-1 >= 0:
      if self.board[cur_move_number][row-1][col] != color_value:
        return False

    if col-1 >= 0:
      if self.board[cur_move_number][row][col-1] != color_value:
        return False

    if row+1 < self.board_size:
      if self.board[cur_move_number][row+1][col] != color_value:
        return False

    if col+1 < self.board_size:
      if self.board[cur_move_number][row][col+1] != color_value:
        return False
    # print('# it is a eye')
    return True

  def wall_review(self, cur_move_number, row, col, color_value):
    cur_index = self.get_next_wall_mark_index()
    self.mark_the_wall(cur_move_number, row+1, col, color_value, cur_index)

    cur_index = self.get_next_wall_mark_index()
    self.mark_the_wall(cur_move_number, row, col+1, color_value, cur_index)

    cur_index = self.get_next_wall_mark_index()
    self.mark_the_wall(cur_move_number, row-1, col, color_value, cur_index)

    cur_index = self.get_next_wall_mark_index()
    self.mark_the_wall(cur_move_number, row, col-1, color_value, cur_index)
    
  def mark_the_wall(self, cur_move_number, row, col, color_value, cur_index):
    if row < 0 or col < 0 or row >= self.board_size or col >= self.board_size:
      # out of board, return
      return

    if self.wall_mark[cur_move_number][row][col] != 0:
      # this stone has been marked, we couldn't overrite it, return
      return
    
    if self.board[cur_move_number][row][col] == color_value:
      self.wall_mark[cur_move_number][row][col] = cur_index

      self.mark_the_wall(cur_move_number, row+1, col, color_value, cur_index)

      self.mark_the_wall(cur_move_number, row, col+1, color_value, cur_index)

      self.mark_the_wall(cur_move_number, row-1, col, color_value, cur_index)

      self.mark_the_wall(cur_move_number, row, col-1, color_value, cur_index)

  def get_wall(self, cur_move_number, row, col, color_value):
    result = set()

    wall_item = self.get_wall_item(cur_move_number, row+1, col, color_value)
    if not wall_item is None:
      result.add(wall_item)

    wall_item = self.get_wall_item(cur_move_number, row, col+1, color_value)
    if not wall_item is None:
      result.add(wall_item)

    wall_item = self.get_wall_item(cur_move_number, row-1, col, color_value)
    if not wall_item is None:
      result.add(wall_item)

    wall_item = self.get_wall_item(cur_move_number, row, col-1, color_value)
    if not wall_item is None:
      result.add(wall_item)

    return list(result)

  def get_wall_item(self, cur_move_number, row, col, color_value):
    if row < 0 or col < 0 or row >= self.board_size or col >= self.board_size:
      # out of board, return
      return None

    return self.wall_mark[cur_move_number][row][col]
     
  
  def get_eye_type(self, cur_move_number, row, col, color_value):
    # detect the type of eye: 0: false eye, 1: true eye, 2: potential true eye

    enemy_color_value = self.reverse_color_value(color_value)

    if row-1 < 0:
      if col-1 < 0:
        if self.board[cur_move_number][row+1][col+1] == color_value:
          return 1
        elif self.board[cur_move_number][row+1][col+1] == 0:
          return 2
        else:
          return 0
      elif col+1 >=self.board_size:
        if self.board[cur_move_number][row+1][col-1] == color_value:
          return 1
        elif self.board[cur_move_number][row+1][col-1] == 0:
          return 2
        else:
          return 0
      else:
        if self.board[cur_move_number][row+1][col-1] == color_value and \
           self.board[cur_move_number][row+1][col+1] == color_value:
          return 1
        elif self.board[cur_move_number][row+1][col-1] == enemy_color_value or \
           self.board[cur_move_number][row+1][col+1] == enemy_color_value:
          return 0
        else:
          return 2
    elif row+1 >= self.board_size:
      if col-1 < 0:
        if self.board[cur_move_number][row-1][col+1] == color_value:
          return 1
        elif self.board[cur_move_number][row-1][col+1] == 0:
          return 2
        else:
          return 0
      elif col+1 >= self.board_size:
        if self.board[cur_move_number][row-1][col-1] == color_value:
          return 1
        elif self.board[cur_move_number][row-1][col-1] == 0:
          return 2
        else:
          return 0
      else:
        if self.board[cur_move_number][row-1][col-1] == color_value and \
           self.board[cur_move_number][row-1][col+1] == color_value:
          return 1
        elif self.board[cur_move_number][row-1][col-1] == enemy_color_value or \
           self.board[cur_move_number][row-1][col+1] == enemy_color_value:
          return 0
        else:
          return 2
    else:
      if col-1 < 0:
        if self.board[cur_move_number][row-1][col+1] == color_value and \
           self.board[cur_move_number][row+1][col+1] == color_value:
          return 1
        elif self.board[cur_move_number][row-1][col+1] == enemy_color_value or \
           self.board[cur_move_number][row+1][col+1] == enemy_color_value:
          return 0
        else:
          return 2
      elif col+1 >= self.board_size:
        if self.board[cur_move_number][row-1][col-1] == color_value and \
           self.board[cur_move_number][row+1][col-1] == color_value:
          return 1
        elif self.board[cur_move_number][row-1][col-1] == enemy_color_value or \
           self.board[cur_move_number][row+1][col-1] == enemy_color_value:
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
            if self.board[cur_move_number][i][j] == color_value:
              same_color_number = same_color_number + 1
            elif self.board[cur_move_number][i][j] == enemy_color_value:
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
  
  def check_potential_location(self, cur_move_number, pos, target_color_value):
    self.reset_review_record()
    target_state = None
    if target_color_value == 1: 
      target_state = self.black_eye_state[cur_move_number]
    elif target_color_value == 2:
      target_state = self.white_eye_state[cur_move_number]
    else:
      return
    
    (row, col) = pos
    if target_state[row][col] != 2:
      # current location is not a potential eye now, just return
      return

    enemy_color_value = self.reverse_color_value(target_color_value)
    is_true_eye = self.eye_review(cur_move_number, row, col, target_color_value, enemy_color_value, target_state)

    
    for i in range(self.board_size):
      for j in range(self.board_size):
        if self.review_record[i][j] == 1:
          if is_true_eye:
            target_state[i][j] = 1
          else:
            target_state[i][j] = 0
          
      


  def eye_review(self, cur_move_number, row, col, target_color_value, enemy_color_value, target_state):

    
    if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
      # current location is out of board, return False, as board is not a bad thoulder
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
      is_true_eye = self.eye_review(cur_move_number, row-1, col-1, target_color_value, enemy_color_value, target_state)
      if not is_true_eye:
        bad_shoulder_number = bad_shoulder_number + 1

      is_true_eye = self.eye_review(cur_move_number, row-1, col+1, target_color_value, enemy_color_value, target_state)
      if not is_true_eye:
        bad_shoulder_number = bad_shoulder_number + 1

      is_true_eye = self.eye_review(cur_move_number, row+1, col-1, target_color_value, enemy_color_value, target_state)
      if not is_true_eye:
        bad_shoulder_number = bad_shoulder_number + 1

      is_true_eye = self.eye_review(cur_move_number, row+1, col+1, target_color_value, enemy_color_value, target_state)
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
      if self.board[cur_move_number][row][col] == 0:
        return False
      elif self.board[cur_move_number][row][col] == enemy_color_value:
        return False
      elif self.board[cur_move_number][row][col] == target_color_value:
        return True


  def is_my_eye(self, color, pos):
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
    self.review_record = np.zeros((self.board_size, self.board_size), dtype=int)

  def is_my_simple_eye(self, color, pos):
    
    color_value = self.get_color_value(color)
    (row, col) = pos

    if color_value == 1:
      if self.black_eye_state[self.move_number][row][col] == 1:
        return True
      else:
        return False
    elif color_value == 2:
      if self.white_eye_state[self.move_number][row][col] == 1:
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

  def get_empty_space(self):
    return self.get_empty_space_at(self.move_number)

  def get_empty_space_at(self, cur_move_number):
    result = list()
    if cur_move_number >= self.history_length:
      return result
    for i in range(self.board_size):
      for j in range(self.board_size):
        if self.board[cur_move_number][i][j] == 0:
          result.append((i, j))
    return result

  def random_simulate_to_end(self):
    simulate_move_number = self.move_number + 1
    if simulate_move_number >= self.history_length:
      return (0, 0)

    last_move_color_value = self.move_history_color_value
    self.board[simulate_move_number] = self.board[self.move_number]

    empty_space = self.get_empty_space_at(simulate_move_number)




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

    if self.move_number < 1:
      result = result + 'No eye records now.'
    else:
      for i in range(self.board_size - 1, -1, -1):
          line = '# '
          for j in range(0, self.board_size):
                if self.board[self.move_number-1][i][j] == 1:
                  line = line + '*'
                if self.board[self.move_number-1][i][j] == 2:
                  line = line + 'O'
                if self.board[self.move_number-1][i][j] == 0:
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




