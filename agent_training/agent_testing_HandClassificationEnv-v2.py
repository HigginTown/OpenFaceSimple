import gym
import HandClassificationEnv
import numpy as np
from stable_baselines.ppo2 import PPO2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools

ENVIRONMENT = "HandClassificationEnv-v3"
LOAD_DIR = ""
SAVE_SAMPLE_DIR = "data_samples"
TIMESTEPS = int(1e5)


def make_env(env_name=ENVIRONMENT):
    return gym.make(f"{env_name}")


def load_model(load_dir=LOAD_DIR):
    return PPO2.load(load_dir)


def generate_samples(model, env, timesteps=TIMESTEPS, load=False, save=True):
    predictions = []
    actual_ranks = []
    observations = []
    if not load:
        for _ in range(timesteps):
            obs = env.reset()
            rank_class_actual = env.rank_class - 1
            actual_ranks.append(rank_class_actual)
            observations.append(obs)

            prediction = model.predict(obs)[0]
            predictions.append(prediction)
        if save:
            np.save(SAVE_SAMPLE_DIR + "ranks", actual_ranks)
            np.save(SAVE_SAMPLE_DIR + "obs", observations)
            print("Saved data samples")
    if load:
        saved_data = np.load(LOAD_DIR)
        actual_ranks = np.load(LOAD_DIR)[1]
        predictions = [model.predict(obs) for obs in saved_data[0]]

    return predictions, actual_ranks


def create_confusion_matrix():
    model = load_model()
    env = make_env()
    y_pred, y_true = generate_samples(model, env)
    cnf = confusion_matrix(y_true, y_pred)
    print("Classification report: ")
    print(classification_report(y_true=y_true, y_pred=y_pred))
    return cnf


def plot_confusion_matrix(cm, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


np.set_printoptions(precision=2)
cnf_matrix = create_confusion_matrix()
plt.figure()
plot_confusion_matrix(cnf_matrix)
plt.show()
