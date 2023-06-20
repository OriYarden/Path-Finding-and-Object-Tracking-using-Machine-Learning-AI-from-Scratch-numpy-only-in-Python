# Path-Finding-and-Object-Tracking-using-Machine-Learning-AI-from-Scratch-numpy-only-in-Python

Here we have a world (grid); its size is 50 by 50:

![download (22)](https://github.com/OriYarden/Path-Finding-and-Object-Tracking-using-Machine-Learning-AI-from-Scratch-numpy-only-in-Python/assets/137197657/e77b5628-1ba8-4424-8fec-ebc233ad09b0)

Inlcuding diagonal movements, the "best path" for the object (yellow) to reach the target (green) requires 49 actions:

![download (23)](https://github.com/OriYarden/Path-Finding-and-Object-Tracking-using-Machine-Learning-AI-from-Scratch-numpy-only-in-Python/assets/137197657/0db1cd7d-1039-4c8e-99a5-06ae817572cd)

In this example, it took our Neural Network model 28 training iterations to find the "best path" to reach the target.

We can plot performance in terms of the number of actions taken to reach the target over training iterations, where each "training iteration" is a single path to the target on the x-axis:

![download (24)](https://github.com/OriYarden/Path-Finding-and-Object-Tracking-using-Machine-Learning-AI-from-Scratch-numpy-only-in-Python/assets/137197657/301cc0c2-0c6f-4fa1-95ec-01551265688f)

The y-axis is the number of actions in each training path, and we can see a clear learning curve. The first training path included almost 1,000 actions to reach the target:

![download (25)](https://github.com/OriYarden/Path-Finding-and-Object-Tracking-using-Machine-Learning-AI-from-Scratch-numpy-only-in-Python/assets/137197657/abd80ee1-4879-4728-9dc5-68d87a54f70c)


and although there is a bit of noise, each subsequent training path required fewer actions to reach the target.
Here we demonstrate machine learning or artificial intelligence in a world grid search example where the object learns the best path to reach the target in the fewest possible number of actions.

We'll go over in detail how we train our numpy weights matrix representing the possible actions that the object can make at every given location in the world (grid).

Let's start off with the actions themselves (along with labeling the actions):

![image](https://github.com/OriYarden/Path-Finding-and-Object-Tracking-using-Machine-Learning-AI-from-Scratch-numpy-only-in-Python/assets/137197657/4d55ac94-c673-4b31-8256-0d0a24acb4a1)

In the center is the "object" (or "/" in labels); in order for the object to move to a new position in the world (grid), we have to add the row-column actions to the object's current position (also represented as row-columns values). We can also randomize the starting positions of the object and target, but let's use [0, 0] and [49, 49] for this example.
At every location in the grid, there are 8 possible actions (labeled 0 through 7); we'll account for actions that would move the object outside of the world (grid) later.

Given the number of possible actions (8 including diagonal movements, 4 if not including diagonal movements), and a grid size of 50 (and world grid size 50 by 50), we can create a weights matrix by multiplying the square of the grid size and the number of possible actions:

    def init_weights(self):
        return np.reshape(self.add_noise(self.number_of_actions*self.grid_size**2)*0.01, [self.grid_size**2, self.number_of_actions])

Our "add_noise" method will be used when initializing the weights matrix as well as in other methods; it just returns random value(s) or "noise":

    @staticmethod
    def add_noise(n=1):
        return np.random.random(n) if n != 1 else np.random.random(n)[0]

To calculate the fewest number of actions required to reach the target, we'll use the following method:

    def calc_minimum_actions_to_target(self, start, target):
        if not self.diagonal_movements:
            return abs(target[0] - start[0]) + abs(target[1] - start[1])
        d = min(abs(target[1] - start[1]), abs(target[0] - start[0]))
        _d = 0 if abs(target[1] - start[1]) == abs(target[0] - start[0]) else max(abs(target[1] - start[1]), abs(target[0] - start[0])) - d
        return d + _d

Which (when including diagonal movements) is the minimum row-column distance between the starting position and the target position, plus the difference between the maximum row-column distance and the minimum if the row and column distances are not equal.

Prior to the training loop(s), we'll create a paths dictionary which will store the positions of the object as it moves to the target for each training path:

        self.paths = {}
        
And we'll use a method to store_paths:

    def store_path(self, pos, new_path=False):
        if new_path:
            self.paths[str(len(self.paths))] = []
        self.paths[str(len(self.paths) - 1)].append(pos)

Our training iteration begins with a while loop; more specifically, while not self.best_path_to_target(self.paths, self.minimum_actions_to_target):

    @staticmethod
    def best_path_to_target(paths, minimum_actions_to_target):
        return False if not paths or len(paths[str(len(paths) - 1)]) - 1 > minimum_actions_to_target else True

This method returns False if the paths dictionary is empty (obviously), and it returns False if the length of the previous path is greater than the minimum number of actions to reach the target; otherwise, it returns True.

Our outter training while loop creates the world (grid) and places the object at its starting position, and also stores that position:

        while not self.best_path_to_target(self.paths, self.minimum_actions_to_target):
            current_pos = [self.start_pos[0], self.start_pos[1]]
            world_grid = self.create_world_grid(current_pos)

            self.store_path(current_pos, new_path=True)

Now the inner training while loop operates until the object reaches the target:

            while current_pos != self.target_pos:
                new_action, new_pos = self.decide_action(current_pos, world_grid)

                feedback = self.get_feedback(current_pos, new_pos)
                actions_feedback = np.zeros(self.number_of_actions).astype(float)
                actions_feedback[new_action] = feedback + 0.5*self.add_noise()

                weights_feedback = np.reshape(world_grid, [self.grid_size**2, 1])*actions_feedback.transpose()
                self.weights += weights_feedback*learning_rate

                current_pos, world_grid = new_pos, self.create_world_grid(new_pos)
                self.store_path(current_pos)

For each iteration of this while loop, we'll decide a new action (i.e. decide_action method), get feedback from that new action (i.e. get_feedback method), create an actions feedback matrix (i.e. actions_feedback variable), create a weights feedback matrix (i.e. multiply the world grid by the actions feedback matrix), and update the weights matrix (i.e. the weights for actions).
Then, we'll update the current position of the object as the new position along with the new world grid, and store the new position in the paths dictionary.

That was one iteration of the inner while loop, which corresponds to just one action; so our object moved once. It's still not at the target position, so the while loop will continue selecting new actions, updating the weights based on the actions, etc. until it reaches the target.
Once the object reaches the target, we have our first path; given that the actions are at least initially random, the object will begin a new path, and then another, etc. until the "best path" is found--and the outter while loop terminates.

Our only weights are for actions themselves, not the locations of the object nor the location of the target position. We give positive feedback when the object's new position is closer to the target position compared to the previous position, and negative feedback if the object moves farther away from the target position.
We utilize some boolean logic (in list comprehensions) to calculate the error (i.e. feedback), which is negative if the object's actions aren't efficient (i.e. moving non-diagonally when diagonal movements are more appropriate):

    def get_feedback(self, current_pos, new_pos):
        error = sum([abs(self.target_pos[xy] - current_pos[xy]) - abs(self.target_pos[xy] - new_pos[xy]) for xy in range(2)])
        if self.diagonal_movements:
            error += 0.5*sum([self.target_pos[xy] == current_pos[xy] == new_pos[xy] for xy in range(2)]) - 2*sum([self.target_pos[xy] != current_pos[xy] == new_pos[xy] for xy in range(2)]) - 0.5*sum([abs(self.target_pos[xy] - current_pos[xy]) < abs(self.target_pos[xy] - new_pos[xy]) for xy in range(2)])
        return error

In our decide_action method:

    def decide_action(self, current_pos, world_grid):
        actions_weights = np.reshape(world_grid, [1, self.grid_size**2])[0]@self.weights + 0.5*self.add_noise(self.number_of_actions)
        edges = self.check_boundaries(current_pos)
        for _edge in edges if edges is not None else []:
            actions_weights[_edge] -= 100000
        new_action, _new_action = self.get_action(actions_weights)
        new_pos = [current_pos[0] + _new_action[0], current_pos[1] + _new_action[1]]
        return new_action, new_pos

we temporarily reduce the value(s) of the weights that correspond to actions that would move the object outside of the world grid using the check_boundaries method to avoid those actions:

    def check_boundaries(self, current_pos, pos_and=[[0, 2, 4, 5, 6], [1, 3, 5, 6, 7], [0, 3, 4, 6, 7], [1, 2, 4, 5, 7]], pos_or=[[0, 4, 6], [1, 5, 7], [2, 4, 5], [3, 6, 7]], pos=[[0, 2], [1, 3], [0, 3], [1, 2]]):
        for op in ['and', 'or']:
            for e, edges in enumerate([[0, 0], [self.grid_size - 1, self.grid_size - 1], [0, self.grid_size - 1], [self.grid_size - 1, 0]]):
                if eval(f'{current_pos[0]} == {edges[0]} {op} {current_pos[1]} == {edges[1]}'):
                    if op == 'and':
                        return pos_and[e] if self.diagonal_movements else pos[e]
                    return (pos_or[e] if self.diagonal_movements else [e]) if current_pos[0] == edges[0] else (pos_or[e + 2] if self.diagonal_movements else [e + 2])


Below, I included the entire class and its methods:

    import numpy as np
    from matplotlib import pyplot as plt

    class NN:
        def __init__(self, grid_size=10, diagonal_movements=False):
            self.grid_size = grid_size
            self.diagonal_movements = diagonal_movements
            self.number_of_actions = 8 if self.diagonal_movements else 4
            self.weights = self.init_weights()

        @staticmethod
        def print_actions_diagram():
            print('''Actions = [
                [-1, -1], [-1, 0], [-1, 1],
                [0,  -1], object,  [0,  1],
                [1,  -1], [1,  0], [1,  1]
                ]''')
            print('''Labels = [
                [4, 0, 6],
                [2, /, 3],
                [5, 1, 7]
                ]''')

        def train(self, start_pos=None, target_pos=None, random_positions=False, new_weights=False, iterations=None, learning_rate=1.0):
            self.init_positions(start_pos, target_pos, random_positions)
            if new_weights:
                self.weights = self.init_weights()

            self.paths = {}
            while not self.best_path_to_target(self.paths, self.minimum_actions_to_target) if iterations is None else len(self.paths) < iterations:
                current_pos = [self.start_pos[0], self.start_pos[1]]
                world_grid = self.create_world_grid(current_pos)

                self.store_path(current_pos, new_path=True)
                while current_pos != self.target_pos:
                    new_action, new_pos = self.decide_action(current_pos, world_grid)

                    feedback = self.get_feedback(current_pos, new_pos)
                    actions_feedback = np.zeros(self.number_of_actions).astype(float)
                    actions_feedback[new_action] = feedback + 0.5*self.add_noise()

                    weights_feedback = np.reshape(world_grid, [self.grid_size**2, 1])*actions_feedback.transpose()
                    self.weights += weights_feedback*learning_rate

                    current_pos, world_grid = new_pos, self.create_world_grid(new_pos)
                    self.store_path(current_pos)

        @staticmethod
        def add_noise(n=1):
            return np.random.random(n) if n != 1 else np.random.random(n)[0]

        def init_weights(self):
            return np.reshape(self.add_noise(self.number_of_actions*self.grid_size**2)*0.01, [self.grid_size**2, self.number_of_actions])

        def randomize_positions(self):
            self.start_pos = [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
            self.target_pos = [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
            while self.start_pos == self.target_pos:
                self.target_pos = [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]

        def calc_minimum_actions_to_target(self, start, target):
            if not self.diagonal_movements:
                return abs(target[0] - start[0]) + abs(target[1] - start[1])
            d = min(abs(target[1] - start[1]), abs(target[0] - start[0]))
            _d = 0 if abs(target[1] - start[1]) == abs(target[0] - start[0]) else max(abs(target[1] - start[1]), abs(target[0] - start[0])) - d
            return d + _d

        def init_positions(self, start_pos, target_pos, random_positions):
            if not self.__dict__.__contains__('start_pos') or start_pos:
                self.start_pos = start_pos if start_pos is not None and all([min(start_pos) >= 0, max(start_pos) < self.grid_size]) else [0, 0]
            if not self.__dict__.__contains__('target_pos') or target_pos:
                self.target_pos = target_pos if target_pos is not None and all([min(target_pos) >= 0, max(target_pos) < self.grid_size]) else [self.grid_size - 1, self.grid_size - 1]
            if random_positions:
                self.randomize_positions()
            self.minimum_actions_to_target = self.calc_minimum_actions_to_target(self.start_pos, self.target_pos)

        @staticmethod
        def best_path_to_target(paths, minimum_actions_to_target):
            return False if not paths or len(paths[str(len(paths) - 1)]) - 1 > minimum_actions_to_target else True

        @staticmethod
        def get_action(actions_weights, actions=[[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]):
            return np.where(actions_weights == max(actions_weights))[0][0], actions[np.where(actions_weights == max(actions_weights))[0][0]]

        def create_world_grid(self, pos):
            _world_grid = np.zeros((self.grid_size, self.grid_size)).astype(float); _world_grid[pos[0], pos[1]] = 1
            return _world_grid

        def store_path(self, pos, new_path=False):
            if new_path:
                self.paths[str(len(self.paths))] = []
            self.paths[str(len(self.paths) - 1)].append(pos)

        def check_boundaries(self, current_pos, pos_and=[[0, 2, 4, 5, 6], [1, 3, 5, 6, 7], [0, 3, 4, 6, 7], [1, 2, 4, 5, 7]], pos_or=[[0, 4, 6], [1, 5, 7], [2, 4, 5], [3, 6, 7]], pos=[[0, 2], [1, 3], [0, 3], [1, 2]]):
            for op in ['and', 'or']:
                for e, edges in enumerate([[0, 0], [self.grid_size - 1, self.grid_size - 1], [0, self.grid_size - 1], [self.grid_size - 1, 0]]):
                    if eval(f'{current_pos[0]} == {edges[0]} {op} {current_pos[1]} == {edges[1]}'):
                        if op == 'and':
                            return pos_and[e] if self.diagonal_movements else pos[e]
                        return (pos_or[e] if self.diagonal_movements else [e]) if current_pos[0] == edges[0] else (pos_or[e + 2] if self.diagonal_movements else [e + 2])

        def decide_action(self, current_pos, world_grid):
            actions_weights = np.reshape(world_grid, [1, self.grid_size**2])[0]@self.weights + 0.5*self.add_noise(self.number_of_actions)
            edges = self.check_boundaries(current_pos)
            for _edge in edges if edges is not None else []:
                actions_weights[_edge] -= 100000
            new_action, _new_action = self.get_action(actions_weights)
            new_pos = [current_pos[0] + _new_action[0], current_pos[1] + _new_action[1]]
            return new_action, new_pos

        def get_feedback(self, current_pos, new_pos):
            error = sum([abs(self.target_pos[xy] - current_pos[xy]) - abs(self.target_pos[xy] - new_pos[xy]) for xy in range(2)])
            if self.diagonal_movements:
            error += 0.5*sum([self.target_pos[xy] == current_pos[xy] == new_pos[xy] for xy in range(2)]) - 2*sum([self.target_pos[xy] != current_pos[xy] == new_pos[xy] for xy in range(2)]) - 0.5*sum([abs(self.target_pos[xy] - current_pos[xy]) < abs(self.target_pos[xy] - new_pos[xy]) for xy in range(2)])
            return error

        def plot_world(self, path=None, show_paths=True):
            if path is None:
                path = len(self.paths) - 1
            _title = f'Training Path #{str(path + 1)}: {int(len(self.paths[str(path)]) - 1)} Actions'
            if not show_paths:
                path = 0
                _title = f'''
                Yellow = Object, Green = Target
                Purple = World, Grid Size: {self.grid_size} x {self.grid_size}
                Best Path to Target: {self.minimum_actions_to_target} Actions'''
            world_grid = np.zeros((self.grid_size, self.grid_size)).astype(int)
            for row, col in self.paths[str(path)] if show_paths else []:
                world_grid[row, col] = 1
            world_grid[self.start_pos[0], self.start_pos[1]] = 3
            world_grid[self.target_pos[0], self.target_pos[1]] = 2
            fig = plt.figure(figsize=(5, 5))
            ax = plt.subplot(1, 1, 1)
            ax.imshow(world_grid)
            ax.set_title(_title)
            ax.axis('off')
            plt.show()

        def plot_performance(self):
            y = [len(v) - 1 for _, v in self.paths.items()]
            fig = plt.figure(figsize=(5, 5))
            ax = plt.subplot(1, 1, 1)
            ax.plot(range(len(y)), y, ls='-', lw=1.5, color=[1, 0, 0])
            ax.plot(range(len(y)), np.ones(len(y))*self.minimum_actions_to_target, ls='--', lw=0.75, color=[0, 0, 0, 0.5])
            ax.set_ylim(-0.1*max(y), max(y) + 0.1*max(y))
            xticks = [int(_xtick) for _xtick in ax.get_xticks() if _xtick == np.round(_xtick, decimals=0) and 0 <= _xtick < len(y)]
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(_xtick) + 1) if xticks[0] == 0 else str(int(_xtick)) for _xtick in ax.get_xticks()])
            _yticks = [int(_ytick) for _ytick in ax.get_yticks() if _ytick >= 0 and int(_ytick) == _ytick]
            _yticks = _yticks[:-abs(len(_yticks) - len(ax.get_yticks()))] if _yticks else np.arange(0, max(y), np.round(max(y) / 4, decimals=0)).astype(int)
            yticks = []
            [yticks.append(self.minimum_actions_to_target) if self.minimum_actions_to_target not in yticks else yticks.append(_ytick) for _ytick in _yticks if _ytick > self.minimum_actions_to_target]
            ax.set_yticks(yticks)
            _ylabel = '''
            Number of Actions
            in Path to Target
            '''
            ax.set_ylabel(_ylabel, fontsize=15, fontweight='bold')
            ax.set_xlabel('Training Paths', fontsize=15, fontweight='bold')
            ax.xaxis.set_label_position('top')
            for _axis in ['x', 'y']:
                ax.tick_params(axis=_axis, which='both', bottom=False if _axis == 'x' else 'on', top=False if _axis == 'y' else 'on', color='gray', labelcolor='gray', labelbottom=False, labeltop=True)
            for _axis in ['top', 'right', 'bottom', 'left']:
                ax.spines[_axis].set_visible(False)
            plt.show()

        def plot_weights_heatmap(self, display_type='default', weights_map={'positive': '_weights[row, col, action] > 0', 'negative': '_weights[row, col, action] < 0', 'best path': '[row, col] in self.paths[str(len(self.paths) - 1)]', 'default': 'True'}, actions=[[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]], _title='Actions-Weights Heatmap'):
            if not display_type.startswith('default'):
                _title = f'''
                Actions-Weights Heatmap
                ({display_type} actions)
                '''
            _weights = np.reshape(self.weights, [self.grid_size, self.grid_size, self.number_of_actions])
            weights = np.zeros((self.grid_size, self.grid_size)).astype(float)
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    for action in range(self.number_of_actions):
                        _row, _col = row + actions[action][0], col + actions[action][1]
                        if 0 <= _row < self.grid_size and 0 <= _col < self.grid_size and eval(weights_map[display_type]):
                            weights[_row, _col] += _weights[row, col, action] if display_type.startswith('default') else abs(_weights[row, col, action])
            fig = plt.figure(figsize=(5, 5))
            ax = plt.subplot(1, 1, 1)
            ax.imshow(weights)
            ax.set_title(_title, fontsize=15)
            ax.axis('off')
            plt.show()

        def plot_paths_heatmap(self):
            paths = np.zeros((self.grid_size, self.grid_size)).astype(int)
            for positions in self.paths.values():
                for pos in positions:
                    paths[pos[0], pos[1]] += 1
            fig = plt.figure(figsize=(5, 5))
            ax = plt.subplot(1, 1, 1)
            ax.imshow(paths)
            ax.set_title('Paths Heatmap', fontsize=15)
            ax.axis('off')
            plt.show()

    nn = NN(diagonal_movements=True, grid_size=50)
    nn.print_actions_diagram()
    nn.train()
    nn.plot_world(show_paths=False)
    nn.plot_world()
    nn.plot_performance()

This includes features such as starting with randomized positions, including or not including diagonal movements, plots, etc.

For example, here is a heatmap of the paths the object took during training that I think looks pretty cool:

![download (26)](https://github.com/OriYarden/Path-Finding-and-Object-Tracking-using-Machine-Learning-AI-from-Scratch-numpy-only-in-Python/assets/137197657/5a825a6b-baf2-4bad-91b8-294508a7ce23)


