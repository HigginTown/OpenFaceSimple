import time
import warnings
from stable_baselines import A2C, PPO2
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
import gym
import HandMakerEnv
import OpenFaceSimpleEnv
import HandClassificationEnv
import re

# filter warnings
warnings.filterwarnings('ignore')

# agent and training meta
POLICY = MlpPolicy
POLICY_NAME = 'MlpPolicy'

ENVIRONMENT = "HandClassificationEnv-v2"
TIMESTEPS = 300000
NETWORK_ARCH = [160, 160]
LEARNING_RATE = 0.001
LOG_INTERVAL = 500
NUM_ENVS = 8  # some algorithms can be run in parallel

START_TIME = time.asctime().replace(' ', '-').replace(':', '-')
TENSORBOARD_DIR = f"logs/tb/"
ALGO_NAME = str(PPO2)
ALGO = re.findall("\'(.*?)\'", ALGO_NAME)[0].split('.')[-1]
MODEL_DIR = f'models/{START_TIME}-{ENVIRONMENT}-{TIMESTEPS}'
TB_LOG_NAME = f"{ALGO}-{TIMESTEPS}-{LEARNING_RATE}-{len(NETWORK_ARCH)}-{ENVIRONMENT}"  # include layer number for quick ref
LOAD_MODEL = False
LOAD_DIR = "models/Sat-Apr-25-21-30-12-2020-HandClassificationEnv-v2-500000.zip"


def train(timesteps=TIMESTEPS):
    print(f"[INFO] STARTING TRAINING: {START_TIME} {ENVIRONMENT}-{POLICY_NAME}-{ALGO}")
    print(f"[INFO] NETWORK ARCH {NETWORK_ARCH}")

    # use vectorized environments for the appropriate algorithms for a speed boost
    env = make_vec_env(ENVIRONMENT, NUM_ENVS)
    # the network architecture can be defined above for any policy
    policy_kwargs = dict(net_arch=NETWORK_ARCH)
    model = PPO2(policy=POLICY, env=env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=TENSORBOARD_DIR,
                 n_steps=1,
                 learning_rate=LEARNING_RATE)
    if LOAD_MODEL:
        model.load(load_path=LOAD_DIR)
    print(f"[INFO] Training for TIMESTEPS {TIMESTEPS}")

    model.learn(total_timesteps=timesteps, log_interval=LOG_INTERVAL, tb_log_name=TB_LOG_NAME)  # experiment select
    print("[INFO] Done training")

    model.save(save_path=MODEL_DIR, cloudpickle=False)
    print(f"[INFO] MODEL SAVED TO {MODEL_DIR}")

    return 0


failed = train()
