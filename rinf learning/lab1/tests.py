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
    # Initialize value_function, policy, possible_states, and rewards_gained
    value_function = [0] * ((rows + 1) * (cols + 1))
    policy = [[0.25] * 4 for _ in range((rows + 1) * (cols + 1))]
    possible_states = [(i, j) for i in range(1, rows + 1) for j in range(1, cols + 1)]
    return value_function, policy, possible_states

def bellman_update(current_state, possible_states, passed_states, possible_actions, rewards, rewards_gained):
    # Append current state to passed states and result
    passed_states.append(current_state)
    result.append(current_state)

    rewards_gained.append(rewards[possible_states.index(current_state)])


    # Check if the destination is reached
    if current_state == possible_states[-1]:
        print('You have arrived')
        return

    # Try different moves until all have been attempted
    choices_tried = []
    while len(choices_tried) < len(possible_actions):
        next_choice = random.choice(possible_actions)
        
        # If move was not tried yet, add it to choices_tried
        if next_choice not in choices_tried:
            choices_tried.append(next_choice)

        next_state = (current_state[0] + next_choice[0], current_state[1] + next_choice[1])

        print('Next state:', next_state)
        # Skip invalid states
        if next_state not in possible_states:
            continue

        # If the move is valid and not visited before, proceed
        if next_state not in passed_states:
            print('Next move is valid')
            current_state = next_state
            bellman_update(current_state, possible_states, passed_states, possible_actions, rewards, rewards_gained)
            break

    else:
        print('Dead end reached')

# Read grid information from CSV
file_path = 'grid.csv'
cols, rows, rewards = read_grid_from_csv(file_path)

# Initialize variables
value_function, policy, possible_states = initialize_variables(rows, cols)

# Initialize parameters
passed_through_states = []
gamma = 0.8

# My data
possible_actions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # Left, Up, Right, Down

result = []
rewards_gained = []


# Execute the first episode
bellman_update((1, 1), possible_states, passed_through_states, possible_actions, rewards, rewards_gained)

print(f"Possible states: {possible_states}")
print(f"Updated RESULT: {result}")
print(f"Rewards gained: {sum(rewards_gained)}")

# Create an instance of the Grid class
grid = Grid(dimensions=[cols, rows], gridFromFile='grid.csv')
grid.drawGrid()
