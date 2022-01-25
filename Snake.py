import pygame
import random

RED = (255,0,0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Snake:
    """
    snake object in game.
    """
    def __init__(self, start_pos) -> None:
        """
        initialize a snake with start position.
        """
        self.length = 1
        self.body = [start_pos]    # only have a head point
        self.is_alive = True
        self.direction = 0    # direction number, (0, 1, 2, 3) = (N, W, S, E)
        self.color = BLUE
        self.head_color = BLACK
    
    def move(self, action):
        """
        implement the action.
        """
        action_dict = {
            0: (0, -1),  # up
            1: (-1, 0),  # left
            2: (0, 1),   # down
            3: (1, 0)    # right
        }
        if (self.direction + 2) % 4 == action:
            # if the action is opposite of direction (means back), it is not be allowed, and the action will reverse.
            action = (action + 2) % 4
        point = (self.body[0][0] + action_dict[action][0], self.body[0][1] + action_dict[action][1])    # new head
        self.body = [point] + self.body
        self.direction = action

class Game:
    """
    game object which be implemented by `pygame`
    """
    def __init__(self, grid_size) -> None:
        """
        initialize the game with shape: grid_size * grid_size
        """
        pygame.init()
        self.action_dim = 4
        self.state_dim = 7
        self.is_end = False
        self.score = 0

        self.window_size = (600, 600)
        self.refresh_time = 50
        self.title = ""
        self.event_list = []
        self.grid_size = grid_size
        self.l = self.window_size[0] / (self.grid_size + 2)
        self.start_pos = (int((self.grid_size-1)/2), int((self.grid_size-1)/2))    # center of board

        self.snake = Snake(self.start_pos)
        self.food = self.generate_food()
        self.next_food = None

        self.color_bg = WHITE
        self.color_blk = BLACK
        self.color_food = RED
    
    def get_input(self):
        """
        get player input by keyboard. return the action.

        it could be used in debugging.
        """
        action = None
        for event in self.event_list:
            if event.type == pygame.KEYDOWN:
                if event.key == ord('w'):  
                    action = 0
                elif event.key == ord('a'): 
                    action = 1
                elif event.key == ord('s'):
                    action = 2
                elif event.key == ord('d'):
                    action = 3
        return action
    
    def render(self):
        """
        render the frame of the game.
        """
        self.screen = pygame.display.set_mode(self.window_size)
        self.event_list = pygame.event.get()
        for event in self.event_list:
            if event.type == pygame.KEYDOWN:
                # skip current epsoid
                if event.key == ord('q'):
                    self.is_end = True
                    return
            if event.type == pygame.QUIT:    
                # exit
                self.quit()
                exit()
        pygame.display.set_caption(self.title)
        self.setup_background()
        self.setup_board()
        self.draw_snake()
        self.draw_food()
        pygame.display.update()

    def wait(self):
        """
        sleep between two frames.
        """
        pygame.time.wait(self.refresh_time)

    def step(self, action):
        """
        implement the action.
        """
        self.snake.move(action)
        head = self.snake.body[0]
        reward = 1 / ((head[0] - self.food[0])**2 + (head[1] - self.food[1])**2 + self.grid_size)
        if head == self.food:
            # food be eaten
            self.next_food = self.generate_food()
            self.snake.length += 1
            self.score += 1
            reward = 1
        else:
            self.snake.body.pop()    # tail move
            if head[0] == 0 or head[1] == 0 or head[0] == self.grid_size+1 or head[1] == self.grid_size+1:
                # collided in border
                self.is_end = True
                self.snake.is_alive = False
                reward = -1
            elif self.snake.body.count(head) > 1:
                # collided self
                self.is_end = True
                self.snake.is_alive = False
                reward = -1
        
        done = 0 if self.snake.is_alive else 1

        # when food be eaten, we need return two state with the food which be eaten and new.
        state, state_next = self.get_state()

        return state, state_next, reward, done

    def reset(self):
        """
        reset the game and retuen the state.
        """
        self.is_end = False
        self.score = 0
        self.snake = Snake(self.start_pos)
        self.food = self.generate_food()
        state, state_next = self.get_state()
        return state

    def get_state(self):
        """
        get current state of game. 
        """
        body = self.snake.body
        head = body[0]
        grid_size = self.grid_size
        x_list = []
        y_list = []
        for (x, y) in body:
            if x == head[0] or y == head[1]:
                if x_list.count(x) == 0:
                    x_list.append(x)
                if y_list.count(y) == 0:
                    y_list.append(y)
        x_list.sort()
        y_list.sort()

        if x_list.index(head[0]) > 0:
            left = head[0] - x_list[x_list.index(head[0])-1]
        else: 
            left = head[0]
        if x_list.index(head[0]) < len(x_list)-1:
            right = x_list[x_list.index(head[0])+1] - head[0]  
        else: 
            right = grid_size + 1 - head[0]
        if y_list.index(head[1]) > 0:
            up = head[1] - y_list[y_list.index(head[1])-1] 
        else: 
            up = head[1]
        if y_list.index(head[1]) < len(y_list)-1:
            down = y_list[y_list.index(head[1])+1] - head[1]
        else: 
            down = grid_size + 1 - head[1]
        
        state = [self.food[0]-head[0], self.food[1]-head[1], self.snake.direction, up, left, down, right]
        if self.next_food is not None:
            self.food = self.next_food
            self.next_food = None
        state_next = [self.food[0]-head[0], self.food[1]-head[1], self.snake.direction, up, left, down, right]

        return state, state_next

    '''
    def get_state_(self):
        head = self.snake.body[0]
        around = {
            "b1": (head[0] - 1, head[1] - 1),
            "b2": (head[0] + 0, head[1] - 1),
            "b3": (head[0] + 1, head[1] - 1),
            "b4": (head[0] - 1, head[1] + 0),
            "b6": (head[0] + 1, head[1] + 0),
            "b7": (head[0] - 1, head[1] + 1),
            "b8": (head[0] + 0, head[1] + 1),
            "b9": (head[0] + 1, head[1] + 1)
        }
        for b in around.keys():
            if (around[b] in self.snake.body or 
                    around[b][0] == 0 or around[b][1] == 0 or 
                    around[b][0] == self.grid_size+1 or around[b][1] == self.grid_size+1):
                around[b] = 1
            else:
                around[b] = 0

        state = [self.food[0]-head[0], self.food[1]-head[1], 
                around["b1"], around["b2"], around["b3"], around["b4"], 
                around["b6"], around["b7"], around["b8"], around["b9"]]
        if self.next_food is not None:
            self.food = self.next_food
            self.next_food = None
        state_next = [self.food[0]-head[0], self.food[1]-head[1], 
                around["b1"], around["b2"], around["b3"], around["b4"], 
                around["b6"], around["b7"], around["b8"], around["b9"]]
        return state, state_next
    '''

    def quit(self):
        """
        quit the game.
        """
        pygame.quit()
            
    def setup_background(self):
        """
        setup the background of game frame.
        """
        self.screen.fill(self.color_bg)

    def setup_board(self):
        """
        setup the board of game frame.
        """
        for x in range(self.grid_size+2):
            for y in range(self.grid_size+2):
                rect = (x*self.l, y*self.l, self.l, self.l)
                if x == 0 or y == 0 or x == self.grid_size+1 or y == self.grid_size+1:
                    width = 0
                else:
                    width = 1
                pygame.draw.rect(self.screen, self.color_blk, rect, width)

    def draw_food(self):
        """
        draw the food in frame.
        """
        x, y = self.food
        rect = (x*self.l, y*self.l, self.l, self.l)
        pygame.draw.rect(self.screen, self.color_food, rect, 0)

    def draw_snake(self):
        """
        draw the snake in frame.
        """
        for pt in self.snake.body:
            rect = (pt[0] * self.l, pt[1] * self.l, self.l, self.l)
            pygame.draw.rect(self.screen, self.snake.color, rect, 0)
            if self.snake.body.index(pt) == 0:
                pygame.draw.rect(self.screen, self.snake.head_color, rect, 3)

    def generate_food(self):
        """
        randomly generate a food in empty space.

        return the positon (x, y) of food.
        """
        while True:
            x = random.randint(1, self.grid_size)
            y = random.randint(1, self.grid_size)
            if (x, y) not in self.snake.body:
                break
        return (x, y)

    def set_title(self, str):
        """
        set the title of window.
        """
        self.title = str