import unittest
import treys
import OpenFaceSimpleEnv
from OpenFaceSimpleEnv import convert_bitlist_to_int

print("testing")


class MyTestCase(unittest.TestCase):
    def test_sample_action_space(self):
        env = OpenFaceSimpleEnv.OpenFaceSimpleEnv()
        s = env.action_space.sample()
        assert s in [0, 1]

    def test_observation_space(self):
        env = OpenFaceSimpleEnv.OpenFaceSimpleEnv()
        o = env.observation_space.sample()

    def test_step(self):
        env = OpenFaceSimpleEnv.OpenFaceSimpleEnv()
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)

    def test_reset(self):
        env = OpenFaceSimpleEnv.OpenFaceSimpleEnv()
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)
        obs = env.reset()
        assert len(obs) == 356

    def test_ten_steps(selfs):
        env = OpenFaceSimpleEnv.OpenFaceSimpleEnv()
        for t in range(10):
            action = t % 2
            obs, r, done, info = env.step(action)
        assert done == True or r < 0, f"ERROR: done {done}, r {r}, steps {t}"

    def test_render(self):
        env = OpenFaceSimpleEnv.OpenFaceSimpleEnv()
        env.render()
        done = env.done
        t = 0
        while env.done is False:
            t += 1
            action = t % 2
            obs, r, done, info = env.step(action)
            env.render()
        assert t == 10, f"Continued for too may steps {t}"
        print(f"Rewards: {r}")

    def test_experience_random(self, steps=15):
        env = OpenFaceSimpleEnv.OpenFaceSimpleEnv()
        o = env.observation_space.sample()
        print("Starting random board")
        env.obs = o
        env.render()
        done = False
        for _ in range(steps):
            if done: break
            print("TEST EXPERIENCE STEP {}".format(_))
            print("Player card: ", treys.Card.int_to_pretty_str(convert_bitlist_to_int(env.obs[320:352])))
            action = env.action_space.sample()

            print("Placing into: {}".format(action))
            obs, r, done, info = env.step(action)
            print(r)
            env.render()

    def test_step(self):
        env = OpenFaceSimpleEnv.OpenFaceSimpleEnv()
        for t in range(10):
            action = t % 2
            obs, r, done, info = env.step(action)
            rew = env._get_reward(obs)

    def test_repeated_action(self):
        # what happens when we step through the environment with the same action?
        env = OpenFaceSimpleEnv.OpenFaceSimpleEnv()
        print("\n Repeating \n")
        t = 0
        actions_dict = {0: 0, 1: 0}
        while not env.done:
            print(f"Steps: {t}")
            env.render()
            action = env.action_space.sample()
            actions_dict[action] = actions_dict[action] + 1
            print(actions_dict)
            print(f"attempting: {action}")
            obs, r, done, _ = env.step(action)
            print(f"obs step: {obs[-4:]}")
            print(f"Reward: {r}")
            print(f"Done yet? {env.done}")
            print("\n")
            t += 1
        print(f"final iterations: {t}")


if __name__ == '__main__':
    unittest.main()
