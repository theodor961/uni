import csv
import random
from gridv1 import Grid

def read_grid_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        cols, rows = [int(value) for value in next(csvreader)]
        rewards = [int(value) for value in next(csvreader)]
    return cols, rows, rewards

def initialize_variables(rows, cols):
    value_function = [0] * ((rows + 1) * (cols + 1))
    policy = [[0.25] * 4 for _ in range((rows + 1) * (cols + 1))]
    possible_states = [(i, j) for i in range(1, rows + 1) for j in range(1, cols + 1)]
    return value_function, policy, possible_states

def bellman_update(current_state, possible_states, passed_states, possible_actions, rewards, rewards_gained, policy, gamma):
    passed_states.append(current_state)
    rewards_gained.append(rewards[possible_states.index(current_state)])

    if current_state == possible_states[-1]:
        print('You have arrived')
        return

    best_action = get_best_action(current_state, possible_actions, policy)
    update_policy(current_state, best_action, policy)

    choices_tried = []
    while len(choices_tried) < len(possible_actions):
        next_choice = random.choice(possible_actions)

        if next_choice not in choices_tried:
            choices_tried.append(next_choice)

        next_state = (current_state[0] + next_choice[0], current_state[1] + next_choice[1])

        print('Next state:', next_state)
        if next_state not in possible_states:
            continue

        if next_state not in passed_states:
            print('Next move is valid')
            current_state = next_state
            bellman_update(current_state, possible_states, passed_states, possible_actions, rewards, rewards_gained, policy, gamma)
            break

    else:
        print('Dead end reached')
        update_policy_dead_end(current_state, policy)

def get_best_action(current_state, possible_actions, policy):
    action_probabilities = policy[possible_states.index(current_state)]
    best_action_index = action_probabilities.index(max(action_probabilities))
    return possible_actions[best_action_index]

def update_policy(state, action, policy):
    state_index = possible_states.index(state)
    action_index = possible_actions.index(action)
    policy[state_index][action_index] += 0.1

def update_policy_dead_end(state, policy):
    state_index = possible_states.index(state)
    for action_index in range(len(possible_actions)):
        policy[state_index][action_index] -= 0.1

# Read grid information from CSV
file_path = 'grid.csv'
cols, rows, rewards = read_grid_from_csv(file_path)

# Initialize variables
value_function, policy, possible_states = initialize_variables(rows, cols)

# Initialize parameters
gamma = 0.1

# My data
possible_actions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # Left, Up, Right, Down

result = []
rewards_gained = []
passed_through_states = []

# Execute the first episode
count = 0
while count < 1000:
    count += 1
    result = []
    rewards_gained = []
    passed_through_states = []

    bellman_update((1, 1), possible_states, passed_through_states, possible_actions, rewards, rewards_gained, policy, gamma)

print(f"Possible states: {possible_states}")
print(f"Updated RESULT: {passed_through_states}")
print(f"Rewards gained: {sum(rewards_gained)}")
print(f"Updated Policy: {policy}")

# Create an instance of the Grid class
grid = Grid(dimensions=[cols, rows], gridFromFile='grid.csv')
grid.drawGrid()
