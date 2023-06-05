from absl import app, flags
from tqdm import tqdm
import numpy as np
from stable_baselines3 import A2C
import os

from locomotion.envs import env_builder
from locomotion.envs.gym_envs import a1_gym_env
from locomotion.robots import a1
from locomotion.robots import laikago
from locomotion.robots import robot_config

FLAGS = flags.FLAGS
flags.DEFINE_enum('robot_type', 'A1', ['A1', 'Laikago'], 'Robot Type.')
flags.DEFINE_enum('motor_control_mode', 'Position', ['Torque', 'Position', 'Hybrid'], 'Motor Control Mode.')
flags.DEFINE_bool('on_rack', False, 'Whether to put the robot on rack.')

ROBOT_CLASS_MAP = {
  'A1': a1.A1,
  'Laikago': laikago.Laikago
}

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}

model_dir = "./models/A2C"
log_dir = "./logs"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def main(_):

  robot = ROBOT_CLASS_MAP[FLAGS.robot_type]
  motor_control_mode = MOTOR_CONTROL_MODE_MAP[FLAGS.motor_control_mode]

  # env = env_builder.build_regular_env(robot, motor_control_mode=motor_control_mode, enable_rendering=True, on_rack=FLAGS.on_rack )
  env = a1_gym_env.A1GymEnv(render=True)
  obs = env.reset()

  model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
  # model_pth = f"{model_dir}/29000.zip"
  # model = A2C.load(model_pth, env)
  TIMESTEPS = 1000
  for i in range(1, 30):
      model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
      model.save(f"{model_dir}/{TIMESTEPS*i}")

  # while True:
  #   action = np.load("action.npy")
  #   obs, reward, done, info = env.step(action)

if __name__ == "__main__":
  app.run(main)