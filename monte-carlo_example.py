import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define the environment
# First, we define the environment by specifying the number of states, the number of actions, and the discount factor gamma
num_states = 5
num_actions = 2
gamma = 0.9

# Define the policy
# Next, we define the policy as a numpy array, 
# each row corresponds to a state and each column corresponds to an action.
# The values in the array represent the probabilities of selecting each action in each state.
policy = np.array([[0.4, 0.6], 
                   [0.8, 0.2], 
                   [0.5, 0.5], 
                   [0.1, 0.9], 
                   [0.3, 0.7]])
                   

# Define the value function
# We initialize the value function V to be all zeros.
V = np.zeros(num_states)

# Define the number of episodes to run
# We specify the number of episodes to run Monte-Carlo policy evaluation for
num_episodes = 5000

# Define a function to generate an episode
# We define a function called generate_episode that generates an episode by repeatedly selecting actions according to the policy 
# and sampling rewards from the environment. The function returns three lists: states, actions, and rewards.
def generate_episode(policy):
    states = []
    rewards = []
    actions = []
    state = np.random.randint(num_states)
    while True:
        action = np.random.choice(num_actions, p=policy[state])
        next_state = np.random.randint(num_states)
        reward = np.random.normal(0, 1)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        if len(states) == 10:
            break
    return states, actions, rewards

# Define a function to print a text representation of the state
def print_state(state):
    state_str = ''
    for i in range(num_states):
        if i == state:
            state_str += 'X'
        else:
            state_str += '-'
    print(state_str)


# Define a function to plot the value function
def plot_value_function(V_values):
    fig, axs = plt.subplots(nrows=num_states, ncols=1, figsize=(6, 10))
    for s in range(num_states):
        axs[s].plot(V_values[:, s])
        axs[s].set_title('S{}'.format(s+1))
        axs[s].set_xlabel('Episodes')
        axs[s].set_ylabel('Value function')
    fig.tight_layout()
    plt.show()


# Run Monte-Carlo policy evaluation
# We loop over the specified number of episodes and for each episode, 
# we generate an episode using the policy and compute the returns for each state.
# We then update the value function by taking a weighted average of the previous estimate and the newly computed returns.
value_functions = np.zeros((num_episodes, num_states))
for i in range(num_episodes):
    states, actions, rewards = generate_episode(policy)
    G = 0
    for t in reversed(range(10)):
        G = gamma * G + rewards[t]
        V[states[t]] = V[states[t]] + 1 / (i+1) * (G - V[states[t]])
        print_state(states[t]) # print the state at each time step
    value_functions[i] = V.copy() # save the estimated value function after each episode
    print('Episode', i+1, 'completed') # print a message to indicate the end of the episode


# Print the estimated value function
#Finally, we print the estimated value function.
# Overall, this code shows how to use Monte-Carlo policy evaluation to estimate the value function of a given policy in a simple environment.
#  Note that this is just a toy example and in practice, you would likely use more sophisticated environments and policies.
print('Estimated value function:', V)

# Plot the estimated value function across episodes
plot_value_function(value_functions)
