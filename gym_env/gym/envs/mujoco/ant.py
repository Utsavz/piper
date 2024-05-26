import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from copy import deepcopy

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        posbefore = deepcopy(self.get_body_com("torso"))
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        posafter = self.get_body_com("torso")
        # print(posafter,np.linalg.norm(posafter - posbefore),(xposafter - xposbefore))
        # forward_reward = (xposafter - xposbefore)/self.dt
        forward_reward = 0
        if np.linalg.norm(posafter[:2])>np.linalg.norm(posbefore[:2]):
            forward_reward = np.linalg.norm(posafter[:2] - posbefore[:2])/self.dt

        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        # reward = (xposafter - xposbefore)/(posafter[1] - posbefore[1])
        # reward = np.linalg.norm(posafter - posbefore)

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            position=deepcopy(posafter))

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + np.random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 5
