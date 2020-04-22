import unittest

import treys

from OpenFaceSimpleEnv import OpenFaceSimpleEnv, convert_bitlist_to_int


class MyTestCase(unittest.TestCase):
    def test_sample_action_space(self):
        env = OpenFaceSimpleEnv()
        s = env.action_space.sample()
        assert s in [0, 1]

    def test_observation_space(self):
        env = OpenFaceSimpleEnv()
        o = env.observation_space.sample()

    def test_step(self):
        env = OpenFaceSimpleEnv()
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)

    def test_reset(self):
        env = OpenFaceSimpleEnv()
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)
        obs = env.reset()
        assert len(obs) == 356

    def test_ten_steps(selfs):
        env = OpenFaceSimpleEnv()
        for t in range(10):
            action = t % 2
            obs, r, done, info = env.step(action)
        assert done == True or r < 0, f"ERROR: done {done}, r {r}, steps {t}"

    def test_render(self):
        env = OpenFaceSimpleEnv()
        env.render()
        done = env.done
        t = 0
        while (t < 20 and env.done == False):
            t += 1
            action = t % 2
            obs, r, done, info = env.step(action)
            env.render()
        assert t == 10, f"Continued for too may steps {t}"
        print(f"Rewards: {r}")

    def test_experience_random(self, steps=15):
        env = OpenFaceSimpleEnv()
        o = env.observation_space.sample()
        print("Starting board")
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
        env = OpenFaceSimpleEnv()
        for t in range(10):
            action = t % 2
            obs, r, done, info = env.step(action)
            print('conv', convert_bitlist_to_int(obs[-4:]))
            print(r)
            # assert convert_bitlist_to_int(obs[-4:]) == r
            rew = env._get_reward(obs)
            print(rew)


if __name__ == '__main__':
    unittest.main()
