import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntObstaclesBigEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.count = 0
        self.mx = 25
        self.my = 30
        self.realgoal = np.array([0,1])
        mujoco_env.MujocoEnv.__init__(self, 'ant_obstaclesbig.xml', 5)
        utils.EzPickle.__init__(self)
        self.randomizeCorrect()

    def randomizeCorrect(self):
        self.realgoal = np.array([self.np_random.choice([0, 1]), self.np_random.choice([0, 1])])
        # 0 = obstacle. 1 = no obstacle.
        # self.realgoal = 0

    def step(self, a):
        self.count += 1

        if self.count % 300 == 0:
            n_qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
            n_qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
            n_qpos[:2] = self.data.qpos[:2]
            self.set_state(n_qpos, n_qvel)

        if np.sum(np.square(self.data.qpos[:2] - np.array([25,30]))) < 15*15:
            self.mx += np.sign(self.data.qpos[0] - self.mx)
            self.my += np.sign(self.data.qpos[1] - self.my)

        else:
            self.mx = 25
            self.my = 30

        # print(np.square(self.data.qpos[:2] - np.array([0,20])))

        n_qpos = np.copy(self.data.qpos[:])
        n_qpos[-2:] = np.array([self.mx,self.my])
        self.set_state(n_qpos, self.data.qvel[:])
        self.do_simulation(a, self.frame_skip)

        # reward = - np.square(np.sum(self.data.qpos[:2] - np.array([35, -35]))) / 100000
        #
        # print(np.square(np.sum(self.data.qpos[:2] - np.array([50,50]))))

        # print(self.data.qpos[:2,0])
        # print(np.array([35,-35]))
        # print(np.square(self.data.qpos[:2, 0] - np.array([35,-35])))
        done = False

        if np.sum(np.square(self.data.qpos[:2] - np.array([35,-35]))) < 30:
            reward = 100
            done = True
        else:
            reward = -0.1
        # print(reward)

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        # return np.concatenate([
        #     self.data.qpos.flat[2:],
        #     self.data.qvel.flat,
        # ])
        # return np.concatenate([
        #     self.data.qpos.flat,
        #     self.data.qvel.flat,
        # ])
        return np.concatenate([
            self.sim.data.qpos.flat[2:-2],
            self.sim.data.qvel.flat[:-2],
            np.clip(self.sim.data.cfrc_ext[:-1], -1, 1).flat,
        ])

    def reset_model(self):
        self.count = 0
        self.mx = 25
        self.my = 30
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.2
