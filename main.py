import psutil
from LolEnvironement import LoLGame
import numpy as np
import pygetwindow as gw
import pyautogui
import time

def is_lol_running():
    # Check if 'League of Legends.exe' is running
    for process in psutil.process_iter(['name']):
        if process.info['name'] == 'League of Legends.exe':
            return True
    return False

def bring_lol_to_focus():
    try:
        # Find the LoL window by its title
        print('Bringing LoL window to focus...')
        print(gw.getAllTitles())
        lol_window = gw.getWindowsWithTitle('League of Legends')[0]
        # If LoL window is minimized, restore it

        print(lol_window)

        if lol_window.isMinimized:
            lol_window.restore()
    except IndexError:
        # If LoL window is not found, print a message
        print('LoL window not found. Please check the game title.')
    except Exception as e:
        # If there is any other error, print it
        print(e)

env = LoLGame()
Q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])
alpha = 0.5
gamma = 0.95
epsilon = 0.1
num_episodes = 50000

try:
    # Main game loop
    while True:
        # Check if LoL is running
        if not is_lol_running():
            print("No LoL game detected. Waiting...")
            time.sleep(5)  # Wait for 5 seconds before checking again
            continue

        # If LoL is running, bring it to focus
        bring_lol_to_focus()

        # Start the episodes
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
except KeyboardInterrupt:
    print('Training interrupted.')
finally:
    pass    