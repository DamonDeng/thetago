from __future__ import absolute_import
from __future__ import print_function
import copy
import random
from itertools import chain, product
from multiprocessing import Process

from flask import Flask, request, jsonify
from flask.ext.cors import CORS
import numpy as np
from go_core.goboard import GoBoard
from six.moves import range


class HTTPFrontend(object):
    '''
    HTTPFrontend is a simple Flask app served on localhost:8080, exposing a REST API to predict
    go moves.
    '''

    def __init__(self, bot, port=8080):
        self.bot = bot
        self.port = port

    def start_server(self):
        ''' Start Go model server '''
        self.server = Process(target=self.start_service)
        self.server.start()

    def stop_server(self):
        ''' Terminate Go model server '''
        self.server.terminate()
        self.server.join()

    def run(self):
        ''' Run flask app'''
        app = Flask(__name__)
        CORS(app, resources={r"/prediction/*": {"origins": "*"}})
        self.app = app

        @app.route('/dist/<path:path>')
        def static_file_dist(path):
            return open("ui/dist/" + path).read()

        @app.route('/large/<path:path>')
        def static_file_large(path):
            return open("ui/large/" + path).read()

        @app.route('/')
        def home():
            # Inject game data into HTML
            board_init = 'initialBoard = ""' # backup variable
            board = {}
            for row in range(19):
                board_row = {}
                for col in range(19):
                    # Get the cell value
                    cell = str(self.bot.go_board.board.get((col, row)))
                    # Replace values with numbers
                    # Value will be be 'w' 'b' or None
                    cell = cell.replace("None", "0")
                    cell = cell.replace("b", "1")
                    cell = cell.replace("w", "2")
                    # Add cell to row
                    board_row[col] = int(cell) # must be an int
                # Add row to board
                board[row] = board_row
            board_init = str(board) # lazy convert list to JSON
            
            return open("ui/demoBot.html").read().replace('"__i__"', 'var boardInit = ' + board_init) # output the modified HTML file

        @app.route('/sync', methods=['GET', 'POST'])
        def exportJSON():
            export = {}
            export["hello"] = "yes?"
            return jsonify(**export)

        @app.route('/prediction_b', methods=['GET', 'POST'])
        def next_move_b():
            '''Predict next black move and send to client.

            Parses the move and hands the work off to the bot.
            '''
            
            bot_row, bot_col = self.bot.select_move('b')
            print('Prediction b:')
            print((bot_col, bot_row))

            self.auto_player = 2

            result = {'i': bot_col, 'j': bot_row}
            json_result = jsonify(**result)
            return json_result

        @app.route('/prediction_w', methods=['GET', 'POST'])
        def next_move_w():
            '''Predict next black move and send to client.

            Parses the move and hands the work off to the bot.
            '''
            
            bot_row, bot_col = self.bot.select_move('w')
            print('Prediction w:')
            print((bot_col, bot_row))

            self.auto_player = 1

            result = {'i': bot_col, 'j': bot_row}
            json_result = jsonify(**result)
            return json_result

        @app.route('/reset', methods=['GET', 'POST'])
        def reset():
            # reset the model to init state
            print('Reseting board')
            self.bot.reset_board()
            result = {}
            result["result"] = "Success"
            return jsonify(**result)

        @app.route('/prediction', methods=['GET', 'POST'])
        def next_move():
            '''Predict next move and send to client.

            Parses the move and hands the work off to the bot.
            '''
            content = request.json
            col = content['i']
            row = content['j']
            print('Received move:')
            print((col, row))
            success, position = self.bot.apply_move('b', (row, col))
            
            if success:

                bot_row = position[0]
                bot_col = position[1]
                print('Prediction:')
                print((bot_col, bot_row))
                result = {'i': bot_col, 'j': bot_row}
                json_result = jsonify(**result)
                return json_result
            else:
                result = {}
                result["result"] = "failed"
                return jsonify(**result)

        self.app.run(host='0.0.0.0', port=self.port, debug=True, use_reloader=False)

    



def get_first_valid_move(board, color, move_generator):
    for move in move_generator:
        if move is None or board.is_move_legal(color, move):
            return move
    return None


def generate_in_random_order(point_list):
    """Yield all points in the list in a random order."""
    point_list = copy.copy(point_list)
    random.shuffle(point_list)
    for candidate in point_list:
        yield candidate


def all_empty_points(board):
    """Return all empty positions on the board."""
    empty_points = []
    for point in product(list(range(board.board_size)), list(range(board.board_size))):
        if point not in board.board:
            empty_points.append(point)
    return empty_points


def fill_dame(board):
    status = scoring.evaluate_territory(board)
    # Pass when all dame are filled.
    if status.num_dame == 0:
        yield None
    for dame_point in generate_in_random_order(status.dame_points):
        yield dame_point
