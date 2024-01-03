import csv
import random
from gridv1 import Grid

with open('grid.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    cols, rows = [int(value) for value in next(csvreader)]
    rewards = [int(value) for value in next(csvreader)]

valueFunction = [0 for i in range(1, rows * cols)]
policy = [[0.25, 0.25, 0.25, 0.25] for i in range(1, rows * cols)]
possibleStates = [(i, j) for i in range(1, rows) for j in range(1, cols)]
passedThroughStates = []
gamma = 0.8


#my data
possible_actions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # Left, Up, Right, Down

def bellman_update(state, value_function, rewards, gamma):
    row, col = state
    current_state_index = (row - 1) * cols + (col - 1)
   

    new_value = 0
    for action in possible_actions:
        new_row, new_col = row + action[0], col + action[1]
        if 1 <= new_row <= rows and 1 <= new_col <= cols:
            new_state_index = (new_row - 1) * cols + (new_col - 1)
            reward = rewards[new_state_index]
            new_value += policy[current_state_index][possible_actions.index(action)] * (
                    reward + gamma * value_function[new_state_index])

    return new_value

def execute_episode():
    current_state = random.choice(possibleStates)
    passed_through_states = [current_state]

    max_episode_length = rows * cols  # Adjust as needed
    episode_length = 0

    while True:
        current_state_index = (current_state[0] - 1) * cols + (current_state[1] - 1)
        valueFunction[current_state_index] = bellman_update(
            current_state, valueFunction, rewards, gamma)

        max_action_index, _ = max(enumerate(valueFunction), key=lambda x: x[1])
        policy[current_state_index] = [0] * len(policy[current_state_index])
        policy[current_state_index][max_action_index] = 1

        
        chosen_action = random.choice(possible_actions)
        new_state = (current_state[0] + chosen_action[0], current_state[1] + chosen_action[1])

        if new_state == (rows, cols):  # Reached the last grid item
            break

        if episode_length >= max_episode_length:
            break

        if new_state in passed_through_states:
            break

        passed_through_states.append(new_state)
        current_state = new_state
        episode_length += 1
        print(f'state {current_state}')

# Execute the first episode
execute_episode()

# Print the updated value function and policy
print(f"Updated value function: {valueFunction}")
print(f"Updated policy: {policy}")

# Create an instance of the Grid class
grid = Grid(dimensions=[cols, rows], gridFromFile='grid.csv')
grid.drawGrid()
