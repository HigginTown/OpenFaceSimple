import time
import warnings
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
import gym
import OpenFaceSimpleEnv
import re

# filter warnings
warnings.filterwarnings('ignore')

# agent and training meta
POLICY = MlpPolicy
POLICY_NAME = 'MlpPolicy'

ENVIRONMENT = "OpenFaceSimpleEnv-v1"
TIMESTEPS = 300000
NETWORK_ARCH = [356, 64, 64, 64, 64, 64]
LEARNING_RATE = 0.0007
LOG_INTERVAL = 100
NUM_ENVS = 4

START_TIME = time.asctime().replace(' ', '-').replace(':', '-')
TENSORBOARD_DIR = f'logs/tb/'
ALGO_NAME = str(A2C)
ALGO = re.findall("\'(.*?)\'", ALGO_NAME)[0].split('.')[-1]
MODEL_DIR = f'models/{START_TIME}-{ENVIRONMENT}-{TIMESTEPS}'
TB_LOG_NAME = f"{ALGO}-{TIMESTEPS}-{LEARNING_RATE}-{len(NETWORK_ARCH)}"  # include layer number for quick ref


def train(policy=POLICY, environment=ENVIRONMENT, timesteps=TIMESTEPS, log_interval=LOG_INTERVAL):
    print(f"[INFO] STARTING TRAINING: {START_TIME} {ENVIRONMENT}-{POLICY_NAME}-{ALGO}")
    print(f"[INFO] NETWORK ARCH {NETWORK_ARCH}")

    # configure the environment
    # env = gym.make("OpenFaceSimpleEnv-v1")
    env = make_vec_env("OpenFaceSimpleEnv-v1", NUM_ENVS)
    # Custom MLP policy of two layers of size 32 each with tanh activation function
    policy_kwargs = dict(net_arch=NETWORK_ARCH)
    model = A2C(policy, env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=TENSORBOARD_DIR, n_steps=10,
                learning_rate=LEARNING_RATE)
    print(f"[INFO] Training for TIMESTEPS {TIMESTEPS}")

    model.learn(total_timesteps=timesteps, log_interval=LOG_INTERVAL, tb_log_name=TB_LOG_NAME)  # experiment select
    print("[INFO] Done training")

    model.save(save_path=MODEL_DIR, cloudpickle=False)
    print(f"[INFO] MODEL SAVED TO {MODEL_DIR}")

    return 0

failed = train()
