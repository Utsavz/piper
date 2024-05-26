import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from copy import deepcopy

class PointObstaclesEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.count = 0
        self.mx = 0
        self.my = 20
        self.change_flag1 = 0
        self.change_flag2 = 0
        self.change_flag3 = 0
        self.change_flag4 = 0
        self.dist_flag1 = 0.9
        self.dist_flag2 = 1.1
        # self.dist_flag2 = 0.5
        self.dist_flag3 = 1.1
        # self.dist_flag4 = 1.1
        self.monster1_dead = 0
        self.monster2_dead = 0
        self.monster3_dead = 0

        self.countMonster2 = 30

        self.monster_collided = False
        self.global_monster_collided = False
        self.realgoal = np.array([0,1])
        self.goal = np.array([60., 10., 0.0])
        mujoco_env.MujocoEnv.__init__(self, 'point_obstacles.xml', 5)
        utils.EzPickle.__init__(self)
        self.randomizeCorrect()

    def randomizeCorrect(self):
        self.realgoal = np.array([self.np_random.choice([0, 1]), self.np_random.choice([0, 1])])
        # 0 = obstacle. 1 = no obstacle.
        # self.realgoal = 0

    def step(self, a):
        current_action = (a[0]+1)/2.0
        # current_action = np.clip(0.5 + (current_action-0.5)*5, 0, 1)
        # print((current_action-0.5))
        # print(current_action)
        # print(self.init_qpos)
        # monster_pos = self.get_body_com("monster")
        # monster_pos[1] += 1
        self.count += 1
        # posafter = self.get_body_com("torso")
        # if self.count % 200 == 0:
        #     n_qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        #     n_qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        #     n_qpos[:2] = self.data.qpos[:2]
        #     self.set_state(n_qpos, n_qvel)

        # if np.sum(np.square(self.data.qpos[:2] - np.array([0,20]))) < 100:
        #     self.mx += np.sign(self.data.qpos[0] - self.mx)
        #     self.my += np.sign(self.data.qpos[1] - self.my)
        #     # self.mx = self.data.qpos[0]
        #     # self.my = self.data.qpos[1]

        #     # n_qpos = np.copy(self.data.qpos[:])
        #     # n_qpos[-2:] = np.array([self.mx,self.my])
        #     # self.set_state(n_qpos, self.data.qvel[:])
        #     # print(self.data.qpos[:2],self.mx,self.my)
        # else:
        #     self.mx = 0
        #     self.my = 20

        # print(np.square(self.data.qpos[:2] - np.array([0,20])))
        n_qpos = np.copy(self.data.qpos[:])

        if not self.change_flag1 and n_qpos[4] >= 5:
            self.dist_flag1 = -self.dist_flag1
            self.change_flag1 = 1
        if self.change_flag1 and n_qpos[4] < -25:
            self.dist_flag1 = -self.dist_flag1
            self.change_flag1 = 0

        if not self.change_flag2 and n_qpos[6] >= 5:
            self.dist_flag2 = -self.dist_flag2
            self.change_flag2 = 1
        if self.change_flag2 and n_qpos[6] < -25:
            self.dist_flag2 = -self.dist_flag2
            self.change_flag2 = 0

        if not self.change_flag3 and n_qpos[8] >= 5:
            self.dist_flag3 = -self.dist_flag3
            self.change_flag3 = 1
        if self.change_flag3 and n_qpos[8] < -25:
            self.dist_flag3 = -self.dist_flag3
            self.change_flag3 = 0

        # print(self.change_flag2)
        # print('later', self.dist_flag2)
        # if not self.change_flag4 and n_qpos[10] >= 5:
        #     self.dist_flag4 = -self.dist_flag4
        #     self.change_flag4 = 1
        # if self.change_flag4 and n_qpos[10] < -27:
        #     self.dist_flag4 = -self.dist_flag4
        #     self.change_flag4 = 0
        # for inner_num in [4,6,8,10,12]:
        reward = -1
        # n_qpos[4] += self.dist_flag1*0.2# * np.random.uniform(0,2)
        # n_qpos[6] += self.dist_flag2*0.2# * np.random.uniform(0,2)
        # n_qpos[8] += self.dist_flag3*0.2# * np.random.uniform(0,2)
        # n_qpos[10] += self.dist_flag4# * np.random.uniform(0,2)

        point_pos = self.get_body_com("torso")

        monster_pos1 = self.get_body_com("monster1")
        monster_pos2 = self.get_body_com("monster2")
        monster_pos3 = self.get_body_com("monster3")
        monster_pos4 = self.get_body_com("monster4")
        
        monster_ypos1 = np.round(monster_pos1[1], 3).copy()
        monster_ypos2 = np.round(monster_pos2[1], 3).copy()
        monster_ypos3 = np.round(monster_pos3[1], 3).copy()
        monster_ypos4 = np.round(monster_pos4[1], 3).copy()

        # monster_pos = deepcopy(np.hstack((monster_ypos1, monster_ypos2, monster_ypos3, 0)).ravel())
        # print(deepcopy(np.hstack((monster_ypos1, monster_ypos2, monster_ypos3, 0)).ravel()))
        monster_pos = deepcopy(np.hstack((0, monster_ypos2, 0, 0)).ravel())
        
        scale_norm = 100.
        pos_before = deepcopy(point_pos/scale_norm)
        min_dist = 100

        # Collision with monster
        # if np.linalg.norm(point_pos - monster_pos1) < 5 or np.linalg.norm(point_pos - monster_pos2) < 5 or np.linalg.norm(point_pos - monster_pos3) < 5 or np.linalg.norm(point_pos - monster_pos4) < 5:
        #     n_qpos[:2] = np.array([0, 0])
        # else:
        # current_action = 0.45
        scale_value = 1
        old_value = n_qpos[0]
        n_qpos[0] += scale_value*current_action
        point_pos = self.get_body_com("torso")

        # print(np.linalg.norm(point_pos - np.array([0, 10, 0])) < 10)
        # print(self.countMonster2)
        # print(current_action)
        # print()

        # print(n_qpos[0])
        # if self.countMonster2 and np.linalg.norm(point_pos - np.array([0, 10, 0])) < 7 and np.abs(current_action) > 0.4:
        if np.linalg.norm(point_pos - np.array([-20, 10, 0])) < 7 and np.abs(current_action) > 0.45:
            # self.countMonster2 -= 1
            n_qpos[4] = -10
            n_qpos[0] = 15
        else:
            n_qpos[4] = 0

        if np.linalg.norm(point_pos - np.array([0, 10, 0])) < 7 and np.abs(current_action) < 0.55:
            # self.countMonster2 -= 1
            n_qpos[6] = -10
            n_qpos[0] = 35
        else:
            n_qpos[6] = 0

        if np.linalg.norm(point_pos - np.array([20, 10, 0])) < 7 and np.abs(current_action) > 0.45:
            # self.countMonster2 -= 1
            n_qpos[8] = -10
            n_qpos[0] = 55
        else:
            n_qpos[8] = 0

        # if self.countMonster2 < 30:
        #     self.countMonster2 -= 1

        # if self.countMonster2 <= 0 and np.linalg.norm(point_pos - np.array([0, 10, 0])) > 5:
        #     self.countMonster2 = 30
        # elif self.countMonster2 <= 0:
        #     self.countMonster2 = 1

        # if np.linalg.norm(point_pos - np.array([0, 0, 0])) > 2 and np.linalg.norm(point_pos - monster_pos1) < 5:
        #     self.monster1_dead = 1

        # if np.linalg.norm(point_pos - np.array([0, 0, 0])) > 2 and np.linalg.norm(point_pos - monster_pos2) < 5:
        #     self.monster2_dead = 1

        # if np.linalg.norm(point_pos - np.array([0, 0, 0])) > 2 and np.linalg.norm(point_pos - monster_pos3) < 5:
        #     self.monster3_dead = 1

        # self.monster1_dead = 1
        # # self.monster2_dead = 1
        # self.monster3_dead = 1

        # if not self.monster1_dead:
        #     if n_qpos[0] > 20:
        #         n_qpos[0] = 20

        # if not self.monster2_dead:
        #     if n_qpos[0] > 40:
        #         n_qpos[0] = 40

        # if not self.monster3_dead:
        #     if n_qpos[0] > 60:
        #         n_qpos[0] = 60

        # if self.monster1_dead:
        #     n_qpos[4] = 20
        #     # reward += 0.25

        # if self.monster2_dead:
        #     n_qpos[6] = 20
        #     # reward += 0.25

        # if self.monster3_dead:
        #     n_qpos[8] = 20
        #     # reward += 0.25

        n_qpos[1] = 0
        # print(self.get_body_com("torso")[0])
        # if self.monster1_dead and self.monster2_dead and self.monster3_dead:
        #     # n_qpos[10] = -100
        #     n_qpos[10] += 0.7

        if self.get_body_com("torso")[0] > 24:
            # n_qpos[10] = -100
            n_qpos[10] += 0.7

        # if not self.monster1_dead or not self.monster2_dead or not self.monster3_dead:
        #     if n_qpos[0] > 78:
        #         n_qpos[0] = 78

        #     self.monster_collided = True
        # else:
        #     self.monster_collided = False

        # if (np.linalg.norm(point_pos - np.array([0, 0, 0])) > 2) and self.monster_collided == True:
        #     self.global_monster_collided = True

        # if not self.global_monster_collided:
        #     n_qpos[0] += 1.5*current_action

        # if self.monster_collided == True:
        #     n_qpos[0] -= 1.5

        if n_qpos[0] > 100:
            n_qpos[0] = 100

        if n_qpos[0] < -10:
            n_qpos[0] = -10

        # if n_qpos[9] > 28:
        #     n_qpos[9] = 28

        # n_qpos[-2:] = np.array([self.mx,self.my])
        self.set_state(n_qpos, self.data.qvel[:])
        self.do_simulation(np.zeros(2), self.frame_skip)
        done = False
        point_pos_after = self.get_body_com("torso")
        pos_after = deepcopy(point_pos_after)

        if np.linalg.norm(point_pos[0]/scale_norm - np.array([60/scale_norm])) < 0.5/scale_norm:
            reward = 0
            # done = True
        # else:
        #     reward = -1
        # print(reward)
        # reward += self.compute_reward(point_pos, self.goal.copy())

        # if np.abs(np.linalg.norm(pos_after) - np.linalg.norm(pos_before)) > 1.0:
        #   reward = 0
        # else:
        #   reward = -1

        #
        # print(np.square(np.sum(self.data.qpos[:2] - np.array([30,30]))))

        # if np.sum(np.square(self.data.qpos[:2] - np.array([38,38]))) < 4:
        #     reward = 100
        #     done = True
        # else:
        #     reward = 0
        ob = self._get_obs()
        info = {
            'position': deepcopy(ob['point_pos'][:2]),
            'pos_before': deepcopy(pos_before[:2]),
            'pos_after': deepcopy(ob['posafter'][:2]),
            'monster_pos': deepcopy(ob['extra_obs'])
        }
        return ob, reward, done, info

    def _get_obs(self):
        # return np.concatenate([
        #     self.data.qpos.flat[2:],
        #     self.data.qvel.flat,
        # ])
        # return np.concatenate([
        #     self.data.qpos.flat,
        #     self.data.qvel.flat,
        # ])
        # print(self.sim.data.cfrc_ext)
        scale_norm = 100.
        monster_pos1 = self.get_body_com("monster1")/scale_norm
        monster_pos2 = self.get_body_com("monster2")/scale_norm
        monster_pos3 = self.get_body_com("monster3")/scale_norm
        monster_pos4 = self.get_body_com("monster4")/scale_norm
        point_pos = self.get_body_com("torso")/scale_norm
        posafter = deepcopy(point_pos)
        point_xpos = np.round(point_pos[0], 3).copy()
        monster_ypos1 = np.round(monster_pos1[1], 3).copy()
        monster_ypos2 = np.round(monster_pos2[1], 3).copy()
        monster_ypos3 = np.round(monster_pos3[1], 3).copy()
        monster_ypos4 = np.round(monster_pos4[1], 3).copy()
        # monster_pos = deepcopy(np.hstack((monster_ypos1, monster_ypos2, monster_ypos3, monster_ypos4)).ravel())
        monster_pos = deepcopy(np.hstack((0, monster_ypos2, 0, 0)).ravel())
        # obs_temp = deepcopy(np.hstack((point_xpos, monster_ypos1, monster_ypos2, monster_ypos3, monster_ypos4)).ravel())
        obs_temp = deepcopy(np.array([point_xpos]))
        goal = self.goal/scale_norm
        achieved_goal_temp = point_pos.copy()
        achieved_goal_temp[1] = 10/scale_norm
        return {
            'observation': obs_temp,
            'achieved_goal': achieved_goal_temp,
            'desired_goal': goal.copy(),
            'extra_obs': monster_pos.copy(),
            'point_pos': point_pos.copy(),
            'posafter': posafter.copy()
        }
        return obs_temp

        # return np.concatenate([
        #     self.sim.data.qpos.flat[2:-2],
        #     self.sim.data.qvel.flat[:-2],
        #     np.clip(self.sim.data.cfrc_ext[:-1], -1, 1).flat,
        # ])

    def compute_reward(self, achieved_goal, goal, info=None, reward_type='sparse'):
        scale_norm = 100.
        self.distance_threshold = 0.5/scale_norm
        # Compute distance between goal and the achieved goal.
        # d = np.linalg.norm(achieved_goal[0] - goal[0])
        # d = self.goal_distance(np.array([achieved_goal[0]]), np.array([goal[0]]))
        # reward_type = 'dense'
        d = self.goal_distance(achieved_goal, goal)
        # print(d)
        reward = 0.
        if reward_type == 'sparse':
            reward = -(d > self.distance_threshold).astype(np.float32)
        else:
            reward = -d/100.
        return reward

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def get_direction_taken(self, action, info=None):
        # Normalize lcode from -1 to 1
        # Return speed
        pos_before = info['pos_before'].copy()
        pos_after = info['pos_after'].copy()

        x1, y1 = pos_before
        x2, y2 = pos_after
        return np.abs((x2 - x1)/ 1.5)

        # # degree = np.degrees(np.arctan2( y2, x2 ))
        # degree = np.degrees(np.arctan2( y2 - y1, x2 - x1 ))
        # # x = (y2-y1) / (x2-x1+0.000001)
        # if degree > 0:
        #   slope_chosen = degree
        # else:
        #   slope_chosen = (360 + degree)
        # # Normalize from -1 to 1
        # slope_chosen = (slope_chosen/180.) - 1

        # # reward = ((-1000.*0.1*(np.square(np.linalg.norm(slope_chosen-target))))/20 + 5.)# * 0.2
        # return slope_chosen

    def setIndex(self, index):
        # print('setIndex')
        self.count = 0
        self.mx = 0
        self.my = 20
        self.monster_collided = False
        self.global_monster_collided = False
        self.monster1_dead = 0
        self.monster2_dead = 0
        self.monster3_dead = 0
        self.change_flag1 = 0
        self.change_flag2 = 0
        self.change_flag3 = 0
        self.change_flag4 = 0
        self.dist_flag1 = 0.9
        self.dist_flag2 = 1.1
        # self.dist_flag2 = 0.5
        self.dist_flag3 = 1.1
        # self.dist_flag4 = 1.1
        self.countMonster2 = 30
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_model(self):
        self.count = 0
        self.mx = 0
        self.my = 20
        self.monster_collided = False
        self.global_monster_collided = False
        self.monster1_dead = 0
        self.monster2_dead = 0
        self.monster3_dead = 0
        self.change_flag1 = 0
        self.change_flag2 = 0
        self.change_flag3 = 0
        self.change_flag4 = 0
        self.dist_flag1 = 0.9
        self.dist_flag2 = 1.1
        # self.dist_flag2 = 0.5
        self.dist_flag3 = 1.1
        # self.dist_flag4 = 1.1
        self.countMonster2 = 30
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8
