from go_core.goboard import GoBoard
from gosgf import Sgf_game
from go_core.sequence_goboard import SequenceGoBoard
from go_core.array_goboard import ArrayGoBoard
from data_loader.kgs_zip_reader import KGSZipReader
import numpy as np
import time

def speed_testing():
  reader = KGSZipReader('data')
  sgf_generator = reader.get_generator()

  time_start=time.time() 
  for i in range(100):
    try:
      if i%100 == 0:
        print('current game number: ' + str(i))
      (file_name, sgf_name, sgf_content) = sgf_generator.next()
      target_board = GoBoard(19)
      apply_sgf_content_to(file_name, sgf_name, sgf_content, target_board)
    except StopIteration:
      print('finished all the files')
      break
  time_end=time.time() 
  print ('time for GoBoard: '+ str(time_end-time_start))

  reader = KGSZipReader('data')
  sgf_generator = reader.get_generator()

  time_start=time.time()
  for i in range(100):
    try:
      if i%100 == 0:
        print('current game number: ' + str(i))
      (file_name, sgf_name, sgf_content) = sgf_generator.next()
      target_board = ArrayGoBoard(19)
      apply_sgf_content_to(file_name, sgf_name, sgf_content, target_board)
    except StopIteration:
      print('finished all the files')
      break
  time_end=time.time(); 
  print ('time for ArrayGoBoard: '+ str(time_end-time_start))

  reader = KGSZipReader('data')
  sgf_generator = reader.get_generator()

  time_start=time.time()
  for i in range(100):
    try:
      if i%100 == 0:
        print('current game number: ' + str(i))
      (file_name, sgf_name, sgf_content) = sgf_generator.next()
      target_board = SequenceGoBoard(19)
      apply_sgf_content_to(file_name, sgf_name, sgf_content, target_board)
    except StopIteration:
      print('finished all the files')
      break
  time_end=time.time(); 
  print ('time for SequenceGoBoard: '+ str(time_end-time_start))

  

def apply_sgf_content_to(file_name, sgf_name, sgf_content, go_board):
  sgf = Sgf_game.from_string(sgf_content)

  main_sequence_iter = sgf.main_sequence_iter()

  if sgf.get_handicap() != None and sgf.get_handicap() != 0:
    # print('handling handicap')
    for setup in sgf.get_root().get_setup_stones():
      for move in setup:
        go_board.apply_move('b', move)

  for item in main_sequence_iter:
    color, move = item.get_move()
    if not color is None and not move is None:
      # print('applying move:' + str(color) + '   move:' + str(move))
      go_board.apply_move(color, move)


def testing():
  reader = KGSZipReader('data')
  sgf_generator = reader.get_generator()

  for i in range(100000):
    try:
      if i%100 == 0:
        print('current game number: ' + str(i))
      (file_name, sgf_name, sgf_content) = sgf_generator.next()
      apply_sgf_content(file_name, sgf_name, sgf_content)
    except StopIteration:
      print('finished all the files')
      break




def apply_sgf_content(file_name, sgf_name, sgf_content):
  sgf = Sgf_game.from_string(sgf_content)

  main_sequence_iter = sgf.main_sequence_iter()

  go_board = GoBoard(19)
  # sequence_go_board = SequenceGoBoard(19)
  array_go_board = ArrayGoBoard(19)
  

  if sgf.get_handicap() != None and sgf.get_handicap() != 0:
    # print('handling handicap')
    for setup in sgf.get_root().get_setup_stones():
      for move in setup:
        go_board.apply_move('b', move)
        array_go_board.apply_move('b', move)

  for item in main_sequence_iter:

    color, move = item.get_move()
    if not color is None and not move is None:
      # print('applying move:' + str(color) + '   move:' + str(move))
      go_board.apply_move(color, move)
      array_go_board.apply_move(color, move)

      go_board_result = go_board.get_array_result()
      array_go_board_result = array_go_board.get_array_result()

      if not array_equal(go_board_result, array_go_board_result):
        print('inconsist!!!!!' + file_name)
        print('sgf file: ' + sgf_name)
        print('color:' + str(color))
        print('move: ' + str(move))

        print(go_board)
        print(array_go_board)

def array_equal(first_array, second_array):
  i_range = len(first_array)
  j_range = len(first_array)

  for i in range(i_range):
    for j in range(j_range):
      if not first_array[i][j] == second_array[i][j]:
        return False
  
  return True

# testing()
speed_testing()