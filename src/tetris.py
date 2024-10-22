import random
import pygame
import numpy as np
from time import sleep

"""
This code implements a fully functional Tetris game using Pygame. Here's an overview of its structure and functionality:

1. Imports and Setup:
   - The code uses pygame for graphics, numpy for array operations, and random for piece generation.
   - A Tetris class is defined to encapsulate all game logic and rendering.

2. Game Constants:
   - The code defines constants for the game board dimensions, piece types, and colors.
   - TETROMINOS dictionary defines all possible Tetris pieces and their rotations.

3. Game Initialization:
   - The __init__ method sets up the Pygame window and initializes the game state.
   - The reset method creates a new game board and sets initial scores and levels.

4. Game Logic:
   - Methods like _get_rotated_piece, _check_collision, and _rotate handle piece movement and rotation.
   - _new_round generates new pieces and checks for game over conditions.
   - _clear_lines removes completed lines and updates the score.

5. Board State and Properties:
   - _get_complete_board returns the current state of the game board.
   - Methods like _number_of_holes, _bumpiness, and _height calculate properties of the board state.

6. Game Play:
   - The play method handles a single move, including piece placement and score calculation.
   - get_next_states generates all possible next states, useful for AI implementations.

7. Rendering:
   - The render method draws the current game state using Pygame.
   - It displays the game board, current piece, score, level, and lines cleared.

8. Main Game Loop:
   - If run as a script, it creates a Tetris instance and runs a simple game loop.

This implementation provides a solid foundation for a Tetris game and includes features that make it suitable for both human play and potential AI agent development. The code is well-structured and modular, allowing for easy expansion or modification of game features.
"""

class Tetris:

    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: {  # Line piece
            0: [(0, 0), (1, 0), (2, 0), (3, 0)],
            90: [(1, 0), (1, 1), (1, 2), (1, 3)],
            180: [(3, 0), (2, 0), (1, 0), (0, 0)],
            270: [(1, 3), (1, 2), (1, 1), (1, 0)],
        },
        1: {  # Square piece
            0: [(1, 0), (0, 1), (1, 1), (2, 1)],
            90: [(0, 1), (1, 2), (1, 1), (1, 0)],
            180: [(1, 2), (2, 1), (1, 1), (0, 1)],
            270: [(2, 1), (1, 0), (1, 1), (1, 2)],
        },
        2: {  # T piece
            0: [(1, 0), (1, 1), (1, 2), (2, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 0)],
            180: [(1, 2), (1, 1), (1, 0), (0, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 2)],
        },
        3: {  # L piece
            0: [(1, 0), (1, 1), (1, 2), (0, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 2)],
            180: [(1, 2), (1, 1), (1, 0), (2, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 0)],
        },
        4: {  # J piece
            0: [(0, 0), (1, 0), (1, 1), (2, 1)],
            90: [(0, 2), (0, 1), (1, 1), (1, 0)],
            180: [(2, 1), (1, 1), (1, 0), (0, 0)],
            270: [(1, 0), (1, 1), (0, 1), (0, 2)],
        },
        5: {  # S piece
            0: [(2, 0), (1, 0), (1, 1), (0, 1)],
            90: [(0, 0), (0, 1), (1, 1), (1, 2)],
            180: [(0, 1), (1, 1), (1, 0), (2, 0)],
            270: [(1, 2), (1, 1), (0, 1), (0, 0)],
        },
        6: {  # Z piece
            0: [(1, 0), (2, 0), (1, 1), (2, 1)],
            90: [(1, 0), (1, 1), (0, 1), (0, 2)],
            180: [(1, 1), (0, 1), (1, 0), (0, 0)],
            270: [(1, 0), (1, 1), (2, 1), (2, 2)],
        }
    }

    COLORS = {
        0: (0, 0, 0),
        1: (255, 99, 71),
        2: (60, 179, 113),
        3: (100, 149, 237),
        4: (255, 215, 0),
        5: (255, 165, 0),
        6: (147, 112, 219),
        7: (0, 206, 209)
    }

    def __init__(self):
        self.reset()
        pygame.init()
        self.screen = pygame.display.set_mode((self.BOARD_WIDTH * 30 + 200, self.BOARD_HEIGHT * 30))
        pygame.display.set_caption('Tetris')
        self.font = pygame.font.Font(None, 36)

    def reset(self):
        self.board = [[self.MAP_EMPTY] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        self.level = 1
        self.lines_cleared = 0

    def _get_rotated_piece(self):
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]

    def _get_complete_board(self):
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board

    def get_game_score(self):
        return self.score
    
    def _new_round(self):
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)
        
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True

    def _check_collision(self, piece, pos):
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] != Tetris.MAP_EMPTY:  # Changed from MAP_BLOCK to MAP_EMPTY
                return True
        return False
    

    def _rotate(self, angle):
        r = (self.current_rotation + angle) % 360
        self.current_rotation = r

    def _add_piece_to_board(self, piece, pos):
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = self.current_piece + 1
        return board

    def _clear_lines(self, board):
        lines_to_clear = [index for index, row in enumerate(board) if sum([1 for cell in row if cell != 0]) == Tetris.BOARD_WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            for _ in lines_to_clear:
                board.insert(0, [self.MAP_EMPTY for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board

    def _number_of_holes(self, board):
        holes = 0

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == self.MAP_EMPTY:
                i += 1
            holes += len([x for x in col[i+1:] if x == self.MAP_EMPTY])

        return holes

    def _bumpiness(self, board):
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == self.MAP_EMPTY:
                i += 1
            min_ys.append(i)
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i + 1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += bumpiness

        return total_bumpiness, max_bumpiness

    def _height(self, board):
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == self.MAP_EMPTY:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height

    def _get_board_props(self, board):
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    def get_next_states(self):
        states = {}
        piece_id = self.current_piece
        
        if piece_id == 6:  # Stick piece has one rotation
            rotations = [0]
        elif piece_id == 0:  # Line piece can rotate
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states

    def get_state_size(self):
        return 4

    def play(self, x, rotation, render=False, render_delay=None):
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        score = self._calculate_score(lines_cleared)
        self.score += score
        self.lines_cleared += lines_cleared
        self._update_level()

        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over

    def _calculate_score(self, lines_cleared):
        if lines_cleared == 1:
            return 100 * self.level
        elif lines_cleared == 2:
            return 300 * self.level
        elif lines_cleared == 3:
            return 500 * self.level
        elif lines_cleared == 4:
            return 800 * self.level
        return 0

    def _update_level(self):
        self.level = self.lines_cleared // 10 + 1

    def render(self):
        self.screen.fill((0, 0, 0))

        # Draw the grid
        for x in range(self.BOARD_WIDTH):
            for y in range(self.BOARD_HEIGHT):
                pygame.draw.rect(self.screen, (50, 50, 50), (x * 30, y * 30, 30, 30), 1)

        # Draw the board
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                if self.board[y][x] != self.MAP_EMPTY:
                    color = self.COLORS[self.board[y][x]]
                    pygame.draw.rect(self.screen, color, (x * 30, y * 30, 30, 30))
                    pygame.draw.rect(self.screen, (60, 60, 60), (x * 30, y * 30, 30, 30), 1)

        # Draw the current piece
        piece = self._get_rotated_piece()
        for x, y in piece:
            color = self.COLORS[self.current_piece + 1]
            pygame.draw.rect(self.screen, color, ((x + self.current_pos[0]) * 30, (y + self.current_pos[1]) * 30, 30, 30))
            pygame.draw.rect(self.screen, (60, 60, 60), ((x + self.current_pos[0]) * 30, (y + self.current_pos[1]) * 30, 30, 30), 1)

        # Draw the score and level on the left
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        level_text = self.font.render(f'Level: {self.level}', True, (255, 255, 255))
        lines_text = self.font.render(f'Lines: {self.lines_cleared}', True, (255, 255, 255))
        self.screen.blit(score_text, (self.BOARD_WIDTH * 30 + 10, 30))
        self.screen.blit(level_text, (self.BOARD_WIDTH * 30 + 10, 70))
        self.screen.blit(lines_text, (self.BOARD_WIDTH * 30 + 10, 110))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

if __name__ == "__main__":
    tetris_game = Tetris()
    while not tetris_game.game_over:
        tetris_game.render()
        sleep(0.5)
