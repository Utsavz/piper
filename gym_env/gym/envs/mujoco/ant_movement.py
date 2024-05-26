
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from copy import deepcopy

class AntMovementEnv(mujoco_env.MujocoEnv, utils.EzPickle):
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
        # print(self.sim.data.cfrc_ext)

        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 3


# import numpy as np
# from gym import utils
# from gym.envs.mujoco import mujoco_env

# class AntMovementEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     def __init__(self):
#         self.realgoal = np.array([1,3])
#         mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
#         utils.EzPickle.__init__(self)
#         self.randomizeCorrect()

#     def randomizeCorrect(self):
#         self.realgoal = np.array([self.np_random.choice([1, 3])])
#         # 0 = obstacle. 1 = no obstacle.

#     def step(self, a):
#         # print(self.data.qpos)
#         # xposbefore = self.data.qpos[0,0] if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[1,0]
#         # yposbefore = self.data.qpos[1,0] if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[0,0]
#         xposbefore = self.data.qpos[0]# if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[1]
#         yposbefore = self.data.qpos[1]# if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[0]

#         self.do_simulation(a, self.frame_skip)

#         # xposafter = self.data.qpos[0,0] if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[1,0]
#         # yposafter = self.data.qpos[1,0] if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[0,0]
#         xposafter = self.data.qpos[0]# if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[1]
#         yposafter = self.data.qpos[1]# if (self.realgoal[0] == 0 or self.realgoal[0] == 1) else self.data.qpos[0]

#         forward_reward = (xposafter - xposbefore)/self.dt
#         # if self.realgoal[0] == 1 or self.realgoal[0] == 3:
#             # forward_reward = forward_reward * -1
#         side_reward = np.abs(yposafter) * 0.5
#         ctrl_cost = .1 * np.square(a).sum()
#         reward = forward_reward - ctrl_cost - side_reward
#         done = False
#         ob = self._get_obs()
#         pos_curr = self.data.qpos[:2]
#         return ob, reward, done, dict(forward_reward=forward_reward, ctrl_cost=ctrl_cost, side_reward=side_reward, pos=pos_curr)

#     def _get_obs(self):
#         return np.concatenate([
#             self.data.qpos.flat,
#             self.data.qvel.flat,
#         ])

#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
#         qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
#         self.set_state(qpos, qvel)
#         return self._get_obs()

#     def viewer_setup(self):
#         self.viewer.cam.distance = self.model.stat.extent * 3
#         # self.viewer.cam.trackbodyid = 0
