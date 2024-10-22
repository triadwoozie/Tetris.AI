import pygame
import random

# Define softer colors
COLORS = [
    (0, 0, 0),       # Black (background)
    (255, 99, 71),   # Tomato (Red)
    (60, 179, 113),  # Medium Sea Green (Green)
    (100, 149, 237), # Cornflower Blue (Blue)
    (255, 215, 0),   # Gold (Yellow)
    (255, 165, 0),   # Orange
    (147, 112, 219), # Medium Purple (Purple)
    (0, 206, 209)    # Dark Turquoise (Cyan)
]

# Define Tetris pieces
SHAPES = [
    [[1, 1, 1, 1]],           # I
    [[1, 1, 1], [0, 1, 0]],   # T
    [[1, 1], [1, 1]],         # O
    [[1, 1, 0], [0, 1, 1]],   # S
    [[0, 1, 1], [1, 1, 0]],   # Z
    [[1, 1, 1], [1, 0, 0]],   # L
    [[1, 1, 1], [0, 0, 1]],   # J
]

class Piece:
    def __init__(self, shape):
        self.shape = shape
        self.color = COLORS[SHAPES.index(shape) + 1]
        self.x = 4
        self.y = 0

    def rotate(self):
        self.shape = [list(row) for row in zip(*self.shape[::-1])]

class Tetris:
    def __init__(self):
        self.grid = [[0] * 10 for _ in range(20)]  # Grid size (10 columns, 20 rows)
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.score = 0
        self.level = 1
        self.lines_cleared = 0

    def new_piece(self):
        return Piece(random.choice(SHAPES))

    def collide(self):
        for i, row in enumerate(self.current_piece.shape):
            for j, cell in enumerate(row):
                if cell:
                    if (self.current_piece.x + j < 0 or
                        self.current_piece.x + j >= len(self.grid[0]) or
                        self.current_piece.y + i >= len(self.grid) or
                        self.grid[self.current_piece.y + i][self.current_piece.x + j]):
                        return True
        return False

    def freeze(self):
        for i, row in enumerate(self.current_piece.shape):
            for j, cell in enumerate(row):
                if cell:
                    self.grid[self.current_piece.y + i][self.current_piece.x + j] = self.current_piece.color

    def clear_lines(self):
        new_grid = [row for row in self.grid if any(cell == 0 for cell in row)]
        lines_cleared = 20 - len(new_grid)
        self.lines_cleared += lines_cleared
        self.score += lines_cleared * 100 * self.level
        self.grid = [[0] * len(self.grid[0]) for _ in range(lines_cleared)] + new_grid
        self.level = self.lines_cleared // 10 + 1
        return lines_cleared

    def move(self, dx, dy):
        self.current_piece.x += dx
        self.current_piece.y += dy
        if self.collide():
            self.current_piece.x -= dx
            self.current_piece.y -= dy
            if dy > 0:
                self.freeze()
                self.clear_lines()
                self.current_piece = self.next_piece
                self.next_piece = self.new_piece()
                if self.collide():
                    return False  # Game over
        return True

    def drop(self):
        while self.move(0, 1):
            pass

    def rotate_piece(self):
        self.current_piece.rotate()
        if self.collide():
            self.current_piece.rotate()  # Rotate back if collision occurs

    def get_grid(self):
        grid_with_piece = [row[:] for row in self.grid]
        for i, row in enumerate(self.current_piece.shape):
            for j, cell in enumerate(row):
                if cell:
                    grid_with_piece[self.current_piece.y + i][self.current_piece.x + j] = self.current_piece.color
        return grid_with_piece

class TetrisGame:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()
        self.tetris = Tetris()
        self.block_size = 30
        self.font = pygame.font.Font(None, 36)

    def draw_grid(self):
        grid = self.tetris.get_grid()
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                color = cell if cell else (40, 40, 40)
                pygame.draw.rect(self.screen, color, (x * self.block_size, y * self.block_size, self.block_size, self.block_size))
                pygame.draw.rect(self.screen, (100, 100, 100), (x * self.block_size, y * self.block_size, self.block_size, self.block_size), 1)

    def draw_sidebar(self):
        sidebar_x = 10 * self.block_size + 20
        pygame.draw.rect(self.screen, (30, 30, 30), (sidebar_x, 0, self.width - sidebar_x, self.height))

        # Draw score
        score_text = self.font.render(f'Score: {self.tetris.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (sidebar_x + 10, 20))

        # Draw level
        level_text = self.font.render(f'Level: {self.tetris.level}', True, (255, 255, 255))
        self.screen.blit(level_text, (sidebar_x + 10, 60))

        # Draw lines cleared
        lines_text = self.font.render(f'Lines: {self.tetris.lines_cleared}', True, (255, 255, 255))
        self.screen.blit(lines_text, (sidebar_x + 10, 100))

        # Draw next piece
        next_piece_text = self.font.render('Next Piece:', True, (255, 255, 255))
        self.screen.blit(next_piece_text, (sidebar_x + 10, 160))

        next_piece = self.tetris.next_piece
        for i, row in enumerate(next_piece.shape):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, next_piece.color, 
                                     (sidebar_x + 20 + j * 20, 200 + i * 20, 20, 20))

    def run(self):
        fall_time = 0
        fall_speed = 0.5
        running = True

        while running:
            fall_time += self.clock.get_rawtime()
            self.clock.tick()

            if fall_time / 1000 > fall_speed:
                fall_time = 0
                self.tetris.move(0, 1)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.tetris.move(-1, 0)
                    if event.key == pygame.K_RIGHT:
                        self.tetris.move(1, 0)
                    if event.key == pygame.K_DOWN:
                        self.tetris.move(0, 1)
                    if event.key == pygame.K_UP:
                        self.tetris.rotate_piece()
                    if event.key == pygame.K_SPACE:
                        self.tetris.drop()

            self.screen.fill((0, 0, 0))
            self.draw_grid()
            self.draw_sidebar()
            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    game = TetrisGame()
    game.run()