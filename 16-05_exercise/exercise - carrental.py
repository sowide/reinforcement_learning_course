from functools import wraps
from itertools import product
from time import sleep

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import poisson
from tqdm import tqdm

"""
Supponiamo di avere state=[state_location_1, state_location_2] macchine all'inizio.

Nella prima parte, muoviamo le macchine da una location ad un'altra durante la notte. Quindi avremo next_state=[
next_state_loc_1, next_state_loc2]=[state_location_1+pi(state), state_location2-pi(state)] macchine.

Nella seconda fase, andiamo ad affittare le macchine. quindi avremo rented_cars = [rented_cars_loc_1, rented_cars_loc2] 
e infine, next_state - rented_cars = [next_state_loc1 - rented_cars_loc1, next_state_loc2 - rented_cars_loc2]

Nell'ultima fase, andiamo a contare le macchine che tornano disponibili return_cars = [return_cars_loc1, 
return_cars_loc2] quindi il numero totale di auto sarà final_cars = [final_cars_loc1, final_cars_loc2] = [
next_state_loc1 - rented_cars_loc1 + return_cars_loc1 , next_state_loc2 - rented_cars_loc2 + return_cars_loc2]
"""

# parameters
MAX_CARS = 20
THETA = 1e-4  # soglia che ci dice quando le iterazioni devono finire
GAMMA = None  # TODO: impostare il discount factor
Mu = [2, 3, 4]  # Valori per la distribuzione di poisson
NUM_STATES = None  # TODO: impostare il numero di stati in una singola location
MAX_MOVE = None  # TODO: impostare massimo numero di macchine che possono essere spostate durante la notte
MOVE_COST = None  # TODO: impostare costo per lo spostamento unitario per singola macchina
PROFIT = None  # TODO: impostare profitto per ogni macchina affittata

# Sia la reward che la probabilità sono condizionate dal numero di macchine all'inizio della seconda fase
Reward = None  # TODO:Inizializzare correttamente l'array
# [s'_1, s'_2, next_s_1, next_s_2]
Prob = np.zeros([NUM_STATES, NUM_STATES, NUM_STATES, NUM_STATES])


def init_poisson_prob(poisson_prob):
    """
    Calcoliamo la probabilità di poisson per ogni singolo stato
    """
    for i, m in enumerate(Mu):
        for n in range(NUM_STATES):
            # Ci servono due probabilità, una è la mass probability
            # e l'altra la funzione di survival
            poisson_prob[i, n, 0] = poisson.pmf(n, m)
            poisson_prob[i, n, 1] = poisson.sf(n - 1, m)


PoissonProb = np.zeros([len(Mu), NUM_STATES, 2])
init_poisson_prob(PoissonProb)


def calc_reward_and_prob():
    """
    La ricompensa per il noleggio e la distribuzione delle auto sono indipendenti dalla stato iniziale s
    quindi, per ogni stato, dobbiamo calcolare la ricompensa del noleggio e la diversa distribuzione delle auto
    per ogni singolo stato.

    Scorriamo tutti i possibili valori dei numeri delle auto a noleggio e tutte le auto che possono essere restituite,
    in questo modo otteniamo la ricompensa corrispondente. Poi sommiamo il tutto, fino a ottenere il valore
    atteso della nostra ricompensa e della probabilità di noleggio con le distribuzioni delle auto.
    """

    for state_location_1, state_location_2 in tqdm(product(list(range(NUM_STATES)), list(range(NUM_STATES))),
                                                   desc="Calculate probability and rewards"):
        # Il numero di macchine affittate è limitato dal numero di macchine in ogni location
        for rent_car_loc1, rent_car_loc2 in product(list(range(state_location_1 + 1)),
                                                    list(range(state_location_2 + 1))):
            r = None  # TODO: Calcolare la reward per le macchine prenotate
            prob_rent_loc1 = PoissonProb[1, rent_car_loc1, int(rent_car_loc1 == state_location_1)]
            prob_rent_loc2 = PoissonProb[2, rent_car_loc2, int(rent_car_loc2 == state_location_2)]
            prob_rent_cars = prob_rent_loc1 * prob_rent_loc2

            # TODO: Calcolare il valore atteso della reward per il noleggio
            Reward[None, None] += None

            # Il numero di macchine che possono tornare indietro è limitato dal numero massimo di macchine in ogni
            # location
            return_loc1_bound, return_loc2_bound = NUM_STATES + rent_car_loc1 - state_location_1, NUM_STATES + rent_car_loc2 - state_location_2
            for return_car_loc1, return_car_loc2 in product(list(range(return_loc1_bound)),
                                                            list(range(return_loc2_bound))):
                prob_ret_loc1 = PoissonProb[1, return_car_loc1, int(return_car_loc1 == return_loc1_bound - 1)]
                prob_ret_loc2 = PoissonProb[0, return_car_loc2, int(return_car_loc2 == return_loc2_bound - 1)]

                # Calcola la distribuzione delle auto
                cars_loc_1 = state_location_1 - rent_car_loc1 + return_car_loc1
                cars_loc_2 = state_location_2 - rent_car_loc2 + return_car_loc2
                Prob[
                    cars_loc_1, cars_loc_2, state_location_1, state_location_2] += prob_rent_cars * prob_ret_loc1 * prob_ret_loc2


def get_value(state_location_1, state_location_2, moved_cars):
    """
    Dato uno stato e un'azione, calcoliamo la value function
    """
    next_state_location_1, next_state_location_2 = min(NUM_STATES - 1, int(state_location_1 + moved_cars)), min(
        NUM_STATES - 1, int(state_location_2 - moved_cars))
    v = None
    # TODO: Dobbiamo calcolare la nostra value function. La value function è la ricompensa immediata +
    #  il discount code per gli stati successivi Per ottenere la seconda parte dell'equazione di bellman bisogna
    #  utilizzare la matrice delle value function e moltiplicarla per quella della probabiltà poissoniane che abbiamo
    #  ottenuto e fare successivamente la somma N.B. dobbiamo sottrarre anche le macchine che vengono spostate
    #  durante la notte
    v = None
    return v


# Funzione che andrà a valutare la policy corrente
# calcolerà per ogni stato il nuovo valore della value function dopo l'improve della policy
def policy_eval():
    delta = float('inf')
    count = 0
    while delta > THETA:
        delta = 0
        count += 1
        # TODO: Per ogni stato all'interno del nostro problema andiamo a prendere la value function dopodiché andiamo
        #  a calcolare il nuovo valore della nostra value function utilizzando la funzione get_value(). N.B. per
        #  calcolare la value function c'è bisogno anche di un'azione. Aggiorniamo la nostra value function e poi
        #  andiamo a calcolare delta. Delta è il massimo fra delta stesso e la differenza, in valore assoluto,
        #  della value function che avevamo precedentemente e quella appena calcolata

        for something in None:
            print("TODO")


def policy_improve():
    policy_stable = True
    # TODO: Per ogni stato nella location1 e nella location 2 bisogna considerare ogni azione che viene presa
    #  in considerazione seguendo la policy.
    #  L'azione che possiamo fare è limitata dal numero di macchine disponibili. Considerate le
    #  azioni valide, dobbiamo calcolare la value function della coppia (stato, azione) e prendere quella che
    #  massimizza il risultato, dopodiché dobbiamo aggiornare la nostra policy
    #  N.B. la policy è una matrice
    #  Come ultimo passo dobbiamo verificare che la nostra azione precedente non sia uguale all'azione che massimizza la
    #  reward. Se le azioni differiscono allora impostiamo polcy_stable = False, altrimenti nulla.
    #  N.B se la nostra azione precedente (contenuta nella policy) è uguale a quella estratta con l'argmax, allora vuol
    #  dire che abbiamo raggiunto una policy stabile

    for something in None:
        print("TODO")

    return policy_stable


def policy_iteration():
    """
    Iterazione delle policy
    """
    policy_stable = False
    while not policy_stable:
        policy_eval()
        policy_stable = policy_improve()


# Calcola il valore atteso della ricompensa del noleggio (in tutti gli stati possibili)
# e le distribuzioni delle varie auto
calc_reward_and_prob()
# Inizializziamo una matrice che conterrà la nostra value function per ogni singolo stato
value = None  # TODO: Inizializzare la matrice
# Inizializziamo una matrice che conterrà la nostra policy per ogni singolo stato
policy = None  # TODO: Inizializzare la matrice
# Iniziamo a iterare cercando la miglior policy
policy_iteration()

# Plottiamo una heatmap
df_policy = pd.DataFrame(
    policy[::-1],
    index=list(range(NUM_STATES - 1, -1, -1)),
    columns=list(range(NUM_STATES)),
)

plt.figure(figsize=(10, 9))
sns.heatmap(data=df_policy, vmin=-5, vmax=5, square=True, cmap="Blues_r")
plt.title('Policy (Jack\'s Car Rental)')
plt.xlabel('Cars (Location B)')
plt.ylabel('Cars (Location A)')
plt.show()
