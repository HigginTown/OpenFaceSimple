import time
import warnings
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
import gym
import OpenFaceSimpleEnv

# filter warnings
warnings.filterwarnings('ignore')

# agent and training meta
POLICY = MlpPolicy
POLICY_NAME = 'MlpPolicy'

ENVIRONMENT = "OpenFaceSimpleEnv-v1"
TIMESTEPS = 200000
NETWORK_ARCH = [356, 356, 800, 800, 400]
LOG_INTERVAL = 100

START_TIME = time.asctime().replace(' ', '-').replace(':', '-')
TENSORBOARD_DIR = f'logs/tb/'
MODEL_DIR = f'models/{START_TIME}'


def train(policy=POLICY, environment=ENVIRONMENT, timesteps=TIMESTEPS, log_interval=LOG_INTERVAL):
    print(f"[INFO] STARTING TRAINING: {START_TIME} {ENVIRONMENT}-{POLICY_NAME}-PPO2")
    print(f"[INFO] NETWORK ARCH {NETWORK_ARCH}")

    # configure the environment
    env = gym.make("OpenFaceSimpleEnv-v1")
    # Custom MLP policy of two layers of size 32 each with tanh activation function
    policy_kwargs = dict(net_arch=NETWORK_ARCH)
    model = A2C(policy, env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=TENSORBOARD_DIR, n_steps=10,
                learning_rate=.001)
    print(f"[INFO] Training for TIMESTEPS {TIMESTEPS}")

    model.learn(total_timesteps=timesteps, log_interval=LOG_INTERVAL, tb_log_name=f"select_5_per")  # experiment select
    print("[INFO] Done training")

    model.save(save_path=MODEL_DIR, cloudpickle=False)
    print(f"[INFO] MODEL SAVED TO {MODEL_DIR}")

    return 0

failed = train()
