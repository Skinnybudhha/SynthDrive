import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
import threading
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
from threading import Thread
from tqdm import tqdm

# Adjust this path according to your Carla installation
CARLA_EGG_PATH = 'D:/carla/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64')

# Set up the sys.path to include the Carla egg file
try:
    sys.path.append(glob.glob(CARLA_EGG_PATH)[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 100  # Set the number of episodes
DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

ep_rewards = []  # List to store episode rewards

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.step += 1
            self.writer.flush()

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        # Choose a new spawn point until it does not result in a collision
        while True:
            try:
                self.transform = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
                self.actor_list.append(self.vehicle)
            except RuntimeError:
                continue
            break

        # Other initialization steps...
        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        # Implementation of image processing...
        pass

    def step(self, action):
        # Implementation of the step function...
        pass

class DQNAgent:
    def __init__(self):
        # Initialization of DQNAgent...
        pass

    def create_model(self):
        # Implementation of create_model...
        pass

    def update_replay_memory(self, transition):
        # Implementation of update_replay_memory...
        pass

    def train(self):
        # Implementation of train...
        pass

    def get_qs(self, state):
        # Implementation of get_qs...
        pass

def visualize_thread():
    # Implementation of visualize_thread...
    pass

env = CarEnv()
agent = DQNAgent()

if __name__ == "__main__":
    visualize_thread = threading.Thread(target=visualize_thread)
    visualize_thread.start()

    for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'):
        env.reset()
        done = False

        episode_reward = 0
        step = 1

        current_state = env.front_camera

        while not done:
            # Implementation of the episode loop...
            pass

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        agent.tensorboard.update_stats(reward_avg=episode_reward)

        if episode % AGGREGATE_STATS_EVERY == 0:
            print(f"Episode: {episode}, epsilon: {epsilon}")

    # Set termination flag after training is done
    agent.terminate = True
    visualize_thread.join()


