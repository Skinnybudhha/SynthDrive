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
from tqdm import tqdm
import glob

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

EPISODES = 500  # Set the number of episodes
DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 100  # Update progress after every 100 episodes

ep_rewards = []  # List to store episode rewards

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._train_dir = self.log_dir  # Add the missing attribute _train_dir
        self._train_step = 1  # Add the missing attribute _train_step

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
        self.actor_list = []  # Initialize actor_list here

    def reset(self):
        self.destroy()  # Destroy existing actors and sensors

        self.collision_hist = []

        while True:
            try:
                self.transform = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
                self.actor_list.append(self.vehicle)

                self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
                self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
                self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
                self.rgb_cam.set_attribute("fov", f"110")

                transform = carla.Transform(carla.Location(x=2.5, z=0.7))
                self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
                self.actor_list.append(self.sensor)
                self.sensor.listen(lambda data: self.process_img(data))
                break
            except RuntimeError:
                self.reset()

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)
        if isinstance(event.other_actor, carla.Vehicle):
            if event.other_actor.is_alive:
                event.other_actor.destroy()
                if event.other_actor in self.actor_list:
                    self.actor_list.remove(event.other_actor)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None

    def destroy(self):
        print("Cleaning up actors...")
        for actor in self.actor_list:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"Error while destroying actor: {e}")

    def __del__(self):
        self.destroy()



class DQNAgent:
    def __init__(self, gradient_accumulation_steps=4):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_accumulation_counter = 0

        self.terminate = False

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states, batch_size=PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states, batch_size=PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0,
                       callbacks=[self.tensorboard])

        self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

def visualize_thread(env):
    while True:
        if env.front_camera is not None:
            cv2.imshow("Front Camera", env.front_camera)
            if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                break

if __name__ == "__main__":
    env = CarEnv()
    agent = DQNAgent()

    visualize_thread = threading.Thread(target=visualize_thread, args=(env,))
    visualize_thread.start()

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.reset()
        done = False

        episode_reward = 0
        step = 1

        current_state = env.front_camera

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, 3)
                time.sleep(1 / 30)

            new_state, reward, done, _ = env.step(action)

            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train()

            current_state = new_state
            step += 1

        ep_rewards.append(episode_reward)

        if episode % AGGREGATE_STATS_EVERY == 0:
            print(f"Episode: {episode}, epsilon: {epsilon}")

            # Save the model after every 100 episodes
            agent.model.save(f"models/{MODEL_NAME}_episode_{episode}.h5")

    agent.terminate = True
    visualize_thread.join()

