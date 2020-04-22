import time
import warnings

from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy

# import stable_baselines

# filter warnings
warnings.filterwarnings('ignore')

# agent and training meta
POLICY = MlpPolicy
POLICY_NAME = 'MlpPolicy'
from stable_baselines.common import make_vec_env

ENVIRONMENT = "OpenFaceSimpleEnv-v0"
TIMESTEPS = 100
NETWORK_ARCH = [356, 356, 356]
LOG_INTERVAL = 1

START_TIME = time.asctime().replace(' ', '-').replace(':', '-')
TENSORBOARD_DIR = f'logs/tb/'
MODEL_DIR = f'models/{START_TIME}'


def train(policy=POLICY, environment=ENVIRONMENT, timesteps=TIMESTEPS, log_interval=LOG_INTERVAL):
    print(f"[INFO] STARTING TRAINING: {START_TIME} {ENVIRONMENT}-{POLICY_NAME}-PPO2")
    print(f"[INFO] NETWORK ARCH {NETWORK_ARCH}")

    # configure the environment
    env = make_vec_env(ENVIRONMENT, n_envs=1)
    # Custom MLP policy of two layers of size 32 each with tanh activation function
    policy_kwargs = dict(net_arch=NETWORK_ARCH)
    model = A2C(policy, env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=TENSORBOARD_DIR, n_steps=10)
    print(f"[INFO] Training for TIMESTEPS {TIMESTEPS}")

    model.learn(total_timesteps=timesteps, log_interval=LOG_INTERVAL, tb_log_name=f"{START_TIME}")

    print("[INFO] Done training")

    model.save(save_path=MODEL_DIR, cloudpickle=False)
    print(f"[INFO] MODEL SAVED TO {MODEL_DIR}")

    return model


trained_model = train()
