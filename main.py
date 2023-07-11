import psutil
from LolEnvironement import LoLGame
import numpy as np
import time

def is_lol_running():
    # Check if 'League of Legends.exe' is running
    for process in psutil.process_iter(['name']):
        if process.info['name'] == 'League of Legends.exe':
            return True
    return False

env = LoLGame()
Q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])

alpha = 0.5
gamma = 0.95
epsilon = 0.1
num_episodes = 50000

# Main game loop
while True:
    # Check if LoL is running
    if not is_lol_running():
        print("No LoL game detected. Waiting...")
        time.sleep(5)  # Wait for 5 seconds before checking again
        continue
    else: 
        print("game detected")

    # If LoL is running, start the episodesq
    for i_episode in range(num_episodes):
        state = env.reset()
        for t in range(100):
            action = np.argmax(Q_table[state]) if np.random.uniform(0, 1) > epsilon else env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            old_value = Q_table[state, action]
            next_max = np.max(Q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            Q_table[state, action] = new_value
            state = next_state
            if done:
                break
    print('Training finished.')