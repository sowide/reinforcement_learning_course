import numpy as np
import argparse
import math
from datetime import datetime as dt

WORLD = np.array([
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "G"],
    ["_", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["X", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "#", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["S", "_", "_", "_", "_", "_", "_", "_", "_", "_"]
])
STATES = range(WORLD.size) #TODO: Add comment
WIDTH = 10
grid = np.indices((WIDTH, WIDTH))
STATE2WORLD = [(x, y) for x, y in zip(grid[0].flatten(), grid[1].flatten())]
START = 90  #TODO: Add comment
CHECKPNT = 64  #TODO: Add comment
GOAL = 9  #TODO: Add comment
WALLS = [ 
    11, 12, 13, 14, 15, 16, 17, 18, 19, 50
]
V = 0  
V_MAX = 3 #TODO: Add comment
V_MIN = 0 #TODO: Add comment
_RIGHT = 0; RIGHT = 1; RIGHT_ = 2
_UP = 3; UP = 4; UP_ = 5
_LEFT = 6; LEFT = 7; LEFT_ = 8
ACTIONS = range(9)

CRASH = -10.   #TODO: Add comment
CHECK = 0.   #TODO: Add comment
WIN = 1000.  #TODO: Add comment
STEP = -1. #TODO: Add comment

PI = np.zeros((len(STATES), len(ACTIONS)))  #TODO: Add comment
Q = np.zeros((len(STATES), len(ACTIONS))) #TODO: Add comment


def reset():
    #TODO: Aggiungi un commento, spiegando dettagliatamente cosa fa questa funzione

    global WORLD
    WORLD = np.array([
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "G"],
        ["_", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["X", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "#", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["S", "_", "_", "_", "_", "_", "_", "_", "_", "_"]
    ])
    global V, V_MAX
    V = 0
    V_MAX = 3


def make_greedy(s, epsilon):
    #TODO: Aggiungi un commento, illustrando dettagliatamente cosa fa questa funzione

    global PI
    PI[s, :] = [epsilon / (len(ACTIONS) - 1.)] * len(ACTIONS)

    best_a = 0
    best_q_val = -np.inf
    for i, q_val in enumerate(Q[s, :]):
        if q_val > best_q_val:
            best_q_val = q_val
            best_a = i

    PI[s, best_a] = 1. - epsilon

    assert np.isclose(np.sum(PI[s, :]), 1.)


def choose_action(s, epsilon):
    #TODO: Aggiungi un commento, illustrando dettagliatamente cosa fa questa funzione
    make_greedy(s, epsilon)
    return np.random.choice(ACTIONS, p=PI[s, :]) 


def move(s, a, beta):
    """
    :param beta: prob of no velocity update (environment stochasticity)
    """
    #TODO: Aggiungi un commento, illustrando dettagliatamente cosa fa questa funzione

    # update velocity with probability 1-beta
    global V, V_MAX
    if np.random.random() < 1-beta:
        if a in [_RIGHT, _UP, _LEFT] and V > V_MIN:
            V -= 1
        elif a in [RIGHT_, UP_, LEFT_] and V < V_MAX:
            V += 1

    r_border = range(WIDTH-1, WIDTH**2, WIDTH) 
    l_border = range(0, WIDTH**2, WIDTH)
    t_border = range(WIDTH)  

    units = range(V)
    check = False  # flag to indicate if we visited the checkpoint
    # move RIGHT of V units:
    if a < len(ACTIONS) / 3:
        for i in units:
            WORLD[STATE2WORLD[s+i]] = '>'  # draw my path gradualy in the world
            # crash: reset world and velocities, return to start state
            if s+i in r_border or s+i+1 in WALLS:
                reset()
                return START, CRASH
            # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
            elif s+i+1 == CHECKPNT:
                check = V_MAX != 5
                V_MAX = 5
            # goal: draw where I end up & return
            elif s+i+1 == GOAL:
                WORLD[STATE2WORLD[s+i+1]] = 'O'
                return s+i+1, WIN
        # draw where I end up & return
        WORLD[STATE2WORLD[s+V]] = 'O'
        return (s+V, CHECK) if check else (s+V, STEP)

    # move UP of V units:
    elif a < 2*len(ACTIONS) / 3:
        for i in units:
            WORLD[STATE2WORLD[s-i*WIDTH]] = '|'  # draw my path gradualy in the world
            # crash: reset world and velocities, return to start state
            if s-i*WIDTH in t_border or s-(i+1)*WIDTH in WALLS:
                reset()
                return START, CRASH
            # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
            elif s-(i+1)*WIDTH == CHECKPNT:
                check = V_MAX != 5
                V_MAX = 5
            # goal: draw where I end up & return
            elif s-(i+1)*WIDTH == GOAL:
                WORLD[STATE2WORLD[s-(i+1)*WIDTH]] = 'O'
                return s-(i+1)*WIDTH, WIN
        # nothing special: draw where I end up & return
        WORLD[STATE2WORLD[s-V*WIDTH]] = 'O'
        return (s-V*WIDTH, CHECK) if check else (s-V*WIDTH, STEP)

    # move LEFT of V units:
    elif a < len(ACTIONS):
        for i in units:
            WORLD[STATE2WORLD[s-i]] = '<'  # draw my path gradualy in the world
            # crash: reset world and velocities, return to start state
            if s-i in l_border or s-i-1 in WALLS:
                reset()
                return START, CRASH
            # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
            elif s-i-1 == CHECKPNT:
                check = V_MAX != 5
                V_MAX = 5
            # goal: draw where I end up & return
            elif s-i-1 == GOAL:
                WORLD[STATE2WORLD[s-i-1]] = 'O'
                return s-i-1, WIN
        # draw where I end up & return
        WORLD[STATE2WORLD[s-V]] = 'O'
        return (s-V, CHECK) if check else (s-V, STEP)

    return s, STEP  # should never happen


def main():
    
    n_episodes = 100 #TODO: Aggiungi un commento
    n_step = 3 #TODO: Aggiungi un commento
    gamma = 0.9 #TODO: Aggiungi un commento
    alpha = 0.1 #TODO: Aggiungi un commento
    epsilon = 0.2 #TODO: Aggiungi un commento
    beta = 0.0

    K = 10 #TODO: Aggiungi un commento

    average_steps = []  #TODO: Aggiungi un commento
    average_reward = []  #TODO: Aggiungi un commento
    for k in range(K):  #TODO: Aggiungi un commento
 
        global Q, PI  # restart learning!!
        PI = np.zeros((len(STATES), len(ACTIONS)))  #TODO: Aggiungi un commento
        Q = np.zeros((len(STATES), len(ACTIONS)))  #TODO: Aggiungi un commento

        n_steps = []  #TODO: Aggiungi un commento
        rewards = []  #TODO: Aggiungi un commento

        start = dt.now()
        ep = 0
        while ep < n_episodes:
            print("\nEpisode", ep+1, "/", n_episodes, "...")
            reset()  #TODO: Aggiungi un commento
            steps = 0 #TODO: Aggiungi un commento
            reward = 0 #TODO: Aggiungi un commento

            states = []  #TODO: Aggiungi un commento
            actions = []  #TODO: Aggiungi un commento
            q = []  #TODO: Aggiungi un commento
            pi = [] #TODO: Aggiungi un commento
            sigmas = [1]  #TODO: Aggiungi un commento
            targets = []  #TODO: Aggiungi un commento

            states.append(START)  #TODO: Aggiungi un commento
            a = choose_action(START, epsilon)  #TODO: Aggiungi un commento
            actions.append(a) #TODO: Aggiungi un commento
            q.append(Q[START, a])   #TODO: Aggiungi un commento
            pi.append(PI[START, a]) #TODO: Aggiungi un commento
            T = np.inf

            t = -1
            while True:
                #TODO: Illustrare dettagliatamente quello che succede all'interno del ciclo while
                t += 1
                assert len(actions) == len(q) == len(pi) == len(sigmas)

                if t < T:
                    s_next, r = move(states[t], actions[t], beta)  
                    states.append(s_next)  
                    steps += 1
                    reward += r
                    if s_next == GOAL:
                        T = t+1
                        targets.append(r - q[t])  
                    else:
                        a_next = choose_action(states[t+1], epsilon)  
                        actions.append(a_next)  
                        sig = 1
                        sigmas.append(sig)  
                        q.append(Q[s_next, a_next]) 
                        target = r + sig*gamma*q[t+1] + (1-sig)*gamma*np.sum(PI[s_next, :]*Q[s_next, :]) - q[t]
                        targets.append(target)
                        pi.append(PI[s_next, a_next]) 
                tau = t - n_step + 1
                if tau >= 0:
                    E = 1
                    G = q[tau]
                    for k in range(tau, min(tau+n_step-1, T-1)):
                        G += E*targets[k]
                        E *= gamma*((1-sigmas[k+1])*pi[k+1] + sigmas[k+1])
                    Q[states[tau], actions[tau]] += alpha*(G - Q[states[tau], actions[tau]])  
                    make_greedy(states[tau], epsilon) 
                if tau == T - 1:
                    break
            print(WORLD)
            ep += 1
            n_steps.append(steps)
            rewards.append(reward)

        avg_n_steps = np.average(n_steps)
        print("average number of steps:", avg_n_steps)
        average_steps.append(avg_n_steps)

        avg_reward = np.average(rewards)
        print("average return:", avg_reward)
        average_reward.append(avg_reward)

    print("\nsteps:", average_steps)
    print("steps avg:", np.average(average_steps))
    print( "rewards:", average_reward)
    print( "rewards avg:", np.average(average_reward))


if __name__ == '__main__':
    main()
