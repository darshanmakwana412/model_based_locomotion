from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab import spaces
import numpy as np
import IPython
import time
import sys
from rllab.envs.mujoco import vrep

BIG = 1e6
MAX_VEL = 999999
MAX_FORCE=1000
VREP_DT = 0.01

class RoachEnv(Serializable):

	def __init__(self, *args, **kwargs):

		self.mode = 0 
		'''
		0 = set vel to either +-max vel depending on torque sign 
			torque -2 to 2, but only abs val of that is set 
		1 = set torque to max (1000)
			set velocity 0 to 200
		'''

		self.VREP_DT = VREP_DT
		vrep.simxFinish(-1)
		self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
		# simExtRemoteApiStart(19997)
		if self.clientID == -1:
			sys.exit('Cannot connect')
		#else:
		# 	print('Connected to remote API server')

		# initialize handles
		self.roach_handle = self._get_handle('Chassis')
		FL_motor_handle = self._get_handle('FL_motor')
		BL_motor_handle = self._get_handle('BL_motor')
		FR_motor_handle = self._get_handle('FR_motor')
		BR_motor_handle = self._get_handle('BR_motor')
		self.motor_handles = [FL_motor_handle, BL_motor_handle, FR_motor_handle, BR_motor_handle]

		# streaming
		vrep.simxGetFloatSignal(self.clientID, "sim_time", vrep.simx_opmode_streaming)
		vrep.simxGetObjectPosition(self.clientID, self.roach_handle, -1, vrep.simx_opmode_streaming)
		vrep.simxGetObjectOrientation(self.clientID, self.roach_handle, -1, vrep.simx_opmode_streaming)
		vrep.simxGetObjectVelocity(self.clientID, self.roach_handle, vrep.simx_opmode_streaming)
		for mh in self.motor_handles:
			vrep.simxGetJointPosition(self.clientID, mh, vrep.simx_opmode_streaming)
			vrep.simxGetObjectFloatParameter(self.clientID, mh, vrep.sim_jointfloatparam_velocity, \
											 vrep.simx_opmode_streaming)

		# super(RoachEnv, self).__init__(*args, **kwargs)
		Serializable.__init__(self, *args, **kwargs)

	def _get_handle(self, name):
		"""return the handle of the object
		"""
		errCode, handle = vrep.simxGetObjectHandle(self.clientID, name, vrep.simx_opmode_blocking)
		assert errCode == 0 or errCode == 1, 'error code should be either 0 or 1.'
		return handle

	def _get_abs_pos(self, handle):
		"""return the absolute position (x,y,z) in meters
		"""
		# Specify -1 to retrieve the absolute position.
		errCode, abs_roach_pos = vrep.simxGetObjectPosition(self.clientID, handle, -1, \
															vrep.simx_opmode_buffer)
		assert errCode == 0 or errCode == 1, 'error code should be either 0 or 1.'
		return abs_roach_pos

	def _get_lin_vel(self, handle):
		"""return the linear velocity of the roach (vx, vy, vz)
		"""
		errCode, lin_vel, _ = vrep.simxGetObjectVelocity(self.clientID, handle, \
														 vrep.simx_opmode_buffer)
		assert errCode == 0 or errCode == 1, 'error code should be either 0 or 1.'
		return lin_vel

	def _get_joint_pos(self, handle):
		"""return the joint/motor position:
		   rotation angle for revolute joint, a 1D value,
		   translation amount for prismatic joint.
		"""
		errCode, joint_pos = vrep.simxGetJointPosition(self.clientID, handle, vrep.simx_opmode_buffer)
		assert errCode == 0 or errCode == 1, 'error code should be either 0 or 1.'
		return joint_pos

	def _get_joint_vel(self, handle):
		"""return the joint/motor velocity, a 1D value
		"""
		errCode, joint_vel = vrep.simxGetObjectFloatParameter(self.clientID, handle, \
															  vrep.sim_jointfloatparam_velocity, \
															  vrep.simx_opmode_buffer)
		assert errCode == 0 or errCode == 1, 'error code should be either 0 or 1.'
		return joint_vel

	def _get_abs_ori(self, handle):
		"""return the orientation of the roach (alpha, beta and gamma)
		"""
		errCode, abs_roach_ori = vrep.simxGetObjectOrientation(self.clientID, handle, -1, \
														   vrep.simx_opmode_buffer)
		assert errCode == 0 or errCode == 1, 'error code should be either 0 or 1.'
		return abs_roach_ori

	def _get_ang_vel(self, handle):
		"""return the angular velocity of the roach (dAlpha, dBeta, dGamma)
		"""
		errCode, _, ang_vel = vrep.simxGetObjectVelocity(self.clientID, handle, \
														 vrep.simx_opmode_buffer)
		assert errCode == 0 or errCode == 1, 'error code should be either 0 or 1.'
		return ang_vel

	# actions are a list of action, each of which is a tuple of three elements:
	# left motor input, right motor input, duration
	def step(self, action, collectingInitialData=False):
		# _, start_time = vrep.simxGetFloatSignal(self.clientID, "sim_time", vrep.simx_opmode_buffer)

		if(self.mode==1):

			force=MAX_FORCE
			vrep.simxSetJointForce(self.clientID, self.motor_handles[0], force, vrep.simx_opmode_streaming)
			vrep.simxSetJointForce(self.clientID, self.motor_handles[1], force, vrep.simx_opmode_streaming)
			vrep.simxSetJointForce(self.clientID, self.motor_handles[2], force, vrep.simx_opmode_streaming)
			vrep.simxSetJointForce(self.clientID, self.motor_handles[3], force, vrep.simx_opmode_streaming)

			left_vel = action[0]
			right_vel = action[1]
			vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles[0], left_vel, vrep.simx_opmode_streaming)
			vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles[1], left_vel, vrep.simx_opmode_streaming)
			vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles[2], right_vel, vrep.simx_opmode_streaming)
			vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles[3], right_vel, vrep.simx_opmode_streaming)

		if(self.mode==0):

			left_torque = action[0]
			right_torque = action[1]

			left_vel = MAX_VEL
			if(left_torque<0):
				left_vel*=-1
				left_torque*=-1
			right_vel = MAX_VEL
			if(right_torque<0):
				right_vel*=-1
				right_torque*=-1

			vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles[0], left_vel, vrep.simx_opmode_streaming)
			vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles[1], left_vel, vrep.simx_opmode_streaming)
			vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles[2], right_vel, vrep.simx_opmode_streaming)
			vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handles[3], right_vel, vrep.simx_opmode_streaming)

			vrep.simxSetJointForce(self.clientID, self.motor_handles[0], left_torque, vrep.simx_opmode_streaming)
			vrep.simxSetJointForce(self.clientID, self.motor_handles[1], left_torque, vrep.simx_opmode_streaming)
			vrep.simxSetJointForce(self.clientID, self.motor_handles[2], right_torque, vrep.simx_opmode_streaming)
			vrep.simxSetJointForce(self.clientID, self.motor_handles[3], right_torque, vrep.simx_opmode_streaming)

		##print(" ***** simxSynchronousTrigger: ",vrep.simxSynchronousTrigger(self.clientID))
		vrep.simxSynchronousTrigger(self.clientID)
		ob = self.get_current_obs()
		# reward: velocity of center of mass
		reward = np.linalg.norm(np.array(self._get_lin_vel(self.roach_handle)))
		return Step(ob, reward, False)

	def get_current_obs(self):
		abs_pos = self._get_abs_pos(self.roach_handle)
		lin_vel = self._get_lin_vel(self.roach_handle)
		abs_ori = self._get_abs_ori(self.roach_handle)
		ang_vel = self._get_ang_vel(self.roach_handle)
		joint_pos, joint_vel = [], []
		for mh in self.motor_handles:
			mp = self._get_joint_pos(mh)
			mv = self._get_joint_vel(mh)
			joint_pos.append(mp)
			joint_vel.append(mv)
		ob = abs_pos + abs_ori + joint_pos + lin_vel + ang_vel + joint_vel
		##print("POS: ", abs_pos)
		return np.array(ob).reshape(-1)

	def get_my_sim_state(self):
		return self.get_current_obs()
	#	 my_sim_state=np.squeeze(np.concatenate((self.model.data.qpos, self.model.data.qvel, self.model.data.qacc, self.model.data.ctrl)))
	#	 return my_sim_state


	def reset(self, init_state=None, evaluating=False, returnStartState=False):
		if(init_state==None):
			for mh in self.motor_handles:
				vrep.simxSetJointTargetVelocity(self.clientID, mh, 0, vrep.simx_opmode_streaming)
			vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
			time.sleep(0.5)
			vrep.simxSynchronous(self.clientID, True)	# enable the synchronous operation mode
			vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

			curr_ob = self.get_current_obs()
			if(returnStartState):
				# starting state:  the data needed to reset to initial state
				# get_current_obs: returns the full state of the roach
				starting_state = curr_ob[np.r_[:3,6:6+len(self.motor_handles)]]
				return curr_ob, starting_state
			else:
				return curr_ob
		else:
			print("\n\n ******** ERROR : trying to reset roach to a certain position. but cannot perform this action ***************\n\n ")
			return -1

	@property
	@overrides
	def action_space(self):
		if(self.mode==1):
			lower_bound = np.array([0, 0])
			upper_bound = np.array([200, 200])
		if(self.mode==0):
			lower_bound = np.array([0, 0]) #########################
			upper_bound = np.array([0.3, 0.3])
		return spaces.Box(lower_bound, upper_bound)

	@property
	@overrides
	def observation_space(self):
		shape = self.get_current_obs().shape
		upper_bound = BIG * np.ones(shape)
		return spaces.Box(upper_bound*(-1), upper_bound)

	# get center of mass
	def get_body_com(self, body_name):
		return self._get_abs_pos(self.clientID, self.roach_handle)


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


# TODO: action_space, observation_space