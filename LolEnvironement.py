import gymnasium as gym
import numpy as np

from controller import  A, Z, E, R, D, F, PressKey, ReleaseKey
from PIL import ImageGrab
from screeninfo import get_monitors
import time
import cv2


class LoLGame(gym.Env):
    def __init__(self):
        super(LoLGame, self).__init__()
        self.action_space = gym.spaces.Discrete(6)  # Six actions: A, Z, E, R, D, F
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1080, 1920))            

    def step(self, action):
        # 1. Perform the action in the game
        self._perform_action(action)

        # 2. Get the new state of the game screen
        screen = np.array(ImageGrab.grab(bbox=(0,0, get_monitors()[0].width, get_monitors()[0].height)))
        new_screen = process_img(screen)
            
        # 3. Determine the reward
        reward = self._calculate_reward()
            
        # 4. Determine if the game is done
        done = self._is_done()

        return new_screen, reward, done, {}

    def _perform_action(self, action):
        # Map the action index to the actual game control
        # Like if action is 0, then press 'A', if action is 1, then press 'Z', etc.
        key_mapping = {0: A, 1: Z, 2: E, 3: R, 4: D, 5: F}
        key_to_press = key_mapping[action]
        PressKey(key_to_press)
        time.sleep(0.1)  # adjust this delay as needed
        ReleaseKey(key_to_press)

    def _calculate_reward(self):
        # This is where you should define your reward
        # This can be tricky, as you need to figure out a way to quantify your bot's performance in the game
        # Here is a very simple example:
        reward = 0
        if self._score_has_increased():
            reward = 1
        elif self._has_died():
            reward = -1
        return reward

    def _is_done(self):
        # You need to figure out a way to tell when your game is over
        # Here is a very simple example:
        return self._has_died()
    

    def _has_died(self):
        # Implement a way to check if game is over
        # This might involve checking the screen for certain indicators
        pass

    def _score_has_increased(self):
        # Implement a way to check if score has increased
        # This might involve checking the screen for certain indicators
        pass

    def _reset_game(self):
        # Implement a way to reset the game
        # This might involve sending a command to the game
        pass

    def reset(self):
        self._reset_game()  # This should be a function that resets your game
        screen = np.array(ImageGrab.grab(bbox=(0,0, get_monitors()[0].width, get_monitors()[0].height)))
        initial_state = process_img(screen)
        return initial_state

    

def roi(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    masked = cv2.bitwise_and(img,mask)
    return masked

def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.resize(processed_img, (84, 84))  # example resize

    # Get image size
    height = processed_img.shape[0]
    width = processed_img.shape[1]
    # Define the vertices for the full screen
    vertices = np.array([[0, height], [0, 0], [width, 0], [width, height]], np.int32)
    processed_img = roi(processed_img, [vertices])
    return processed_img