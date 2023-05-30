import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the actions
ACTIONS = ["rock", "paper", "scissors"]

# Create the Q-table for each agent
num_agents = 2
num_actions = len(ACTIONS)
q_tables = #TODO: Inizializza la qtable, ricorda che è una 3x3 e ogni agente deve averne una

# Define the learning rate and discount factor
learning_rate = None #TODO: Definisci un learning rate a piacere
discount_factor = None  #TODO: Definisci un discount factor a piacere

# Define the exploration rate (epsilon)
epsilon = None #TODO: Definisci un epsilon a piacere

# Play the game for a certain number of episodes
num_episodes = None #TODO: Definisci un numero di episodi a piacere

# Store Q-tables for visualization
q_tables_history = [[np.copy(q_table) for q_table in q_tables] for _ in range(10)]

for episode in tqdm(range(num_episodes)):
    # Initialize the state for each agent
    state = [] #TODO: Inizializza lo stato. Estrai un'azione casuale per ogni agente

    # Choose actions for each agent based on the Q-table with epsilon-greedy exploration
    actions = []
    # TODO: Implementa l'epsilon greedy policy function
    # Per ogni agente estrai un numero casuale, se è minore di epsilon allora fai un'azione casuale.
    # Se, invece, è maggiore, allora prendi l'argmax della qtable dell'agente. 
    # Una volta presa la q_tables dell'agente bisogna poi scegliere prendere tutta la colonna corrispondente all'azione (stato) fatta dall'agente in questione

    for i in range(num_agents):
        #TODO
        None

    # Simulate the game and determine the rewards
    opponent_actions = [] #TODO: Per ogni agente estrai un'azione casuale dell'avversario
    rewards = [0, 0]


    if actions[0] == actions[1]:
        rewards = [0, 0]
    elif (actions[0] == 0 and actions[1] == 2) or (actions[0] == 1 and actions[1] == 0) or (actions[0] == 2 and actions[1] == 1):
        rewards = [1, -1]
    else:
        rewards = [-1, 1]

    # Update the Q-tables for each agent
    for i in range(num_agents):
        q_tables[i][state[i], actions[i]] = None #TODO: applicare la formula del Q-Learning
        
        
   # Store Q-tables for visualization every 100 episodes
    if (episode + 1) % (num_episodes//10) == 0:
        q_tables_history[(episode + 1) // (num_episodes//10) - 1] = [np.copy(q_table) for q_table in q_tables]

# Test the learned policies
num_test_episodes = 30

for episode in range(num_test_episodes):
    state = [] #TODO: Inizializza lo stato. Estrai un'azione casuale per ogni agente
    actions = [np.argmax(q_tables[i][state[i], :]) for i in range(num_agents)]

    print("Episode:", episode + 1)
    print("Agent 1 chooses:", ACTIONS[actions[0]])
    print("Agent 2 chooses:", ACTIONS[actions[1]])
    print("---")
    
        # Determine the winner
    if actions[0] == actions[1]:
        print("It's a tie!")
    elif (actions[0] == 0 and actions[1] == 2) or (actions[0] == 1 and actions[1] == 0) or (actions[0] == 2 and actions[1] == 1):
        print("Agent 1 wins!")
    else:
        print("Agent 2 wins!")
    print("---")
    
    
# Visualize the Q-tables
plt.figure(figsize=(12, 8))
for agents in (range(num_agents)):
    for i, q_tables_epoch in enumerate(q_tables_history):
        plt.subplot(2, 5, i + 1)
        plt.title("Epoch " + str((i + 1) * num_episodes//10) + "\nAgent "+ str(agents + 1))
        plt.imshow((q_tables_epoch[agents]).reshape((num_actions, num_actions)), cmap="hot")
        plt.colorbar()

        plt.xticks(np.arange(num_actions), ACTIONS)
        plt.yticks(np.arange(num_actions), ACTIONS)
        plt.xlabel("Agent 1 Action")
        plt.ylabel("Agent 2 Action")

    plt.tight_layout()
    plt.show()



