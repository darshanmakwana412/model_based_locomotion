from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from .mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs


class SwimmerEnv(MujocoEnv, Serializable):

    FILE = 'swimmer.xml'

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(SwimmerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
            self.get_body_comvel("torso").flat,
        ]).reshape(-1)

    def step(self, action, collectingInitialData=False):
        xposbefore = self.model.data.qpos[0,0]
        #xposbefore = self.get_body_com("torso")[0]
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        forward_reward = self.get_body_comvel("torso")[0]

        ########### rllab rewards
        reward = forward_reward - ctrl_cost

        ########### open ai gym rewards
        '''ctrl_cost_coeff = 0.0001
        reward_ctrl = - ctrl_cost_coeff * np.square(action).sum()
        xposafter = self.model.data.qpos[0,0]
        #xposafter = self.get_body_com("torso")[0]
        reward_fwd = (xposafter - xposbefore) / 0.01
        reward = reward_fwd + reward_ctrl'''

        done = False
        return Step(next_obs, reward, done)

    def get_my_sim_state(self):
        my_sim_state=np.squeeze(np.concatenate((self.model.data.qpos, self.model.data.qvel, self.model.data.qacc, self.model.data.ctrl)))
        return my_sim_state

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))