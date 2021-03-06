import pybullet as p
#import time
import math
import numpy as np
from datetime import datetime
from time import sleep
import os
import copy
from numpy import sin, cos, pi
import gym
from gym import core, spaces, error
from gym.utils import seeding
import ABB120
from math3D import *


class env(gym.GoalEnv):
    def __init__(self, render, test=False, max_step=2048):
        self.max_step = max_step
        self.test = test
        J_home = np.array([0,  0,  30,  0,  60,  0])
        n_actions = 6
        max_Joints_offset = 1
        self.dt = 0.1
        
        self.sim = p
        if render:
            self.sim.connect(self.sim.GUI)
        else:
            self.sim.connect(self.sim.DIRECT)
        self.sim.resetDebugVisualizerCamera(  cameraDistance=1, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0.2,0,1])
        
        self.ABB120Id = self.sim.loadURDF("3Dmodels/IRB120/model.urdf", [0, 0, 0.8], useFixedBase=True)
        self.tableId = self.sim.loadURDF("3Dmodels/Table/table.urdf", [0, 0, 0], useFixedBase=True)
        self.point1a = self.sim.addUserDebugLine(np.array([0, 0, 0.8]),np.array([0, 0, 0.8]),[0,1,0])
        self.point2a = self.sim.addUserDebugLine(np.array([0, 0, 0.8]),np.array([0, 0, 0.8]),[1,0,0])
        self.point1b = self.sim.addUserDebugLine(np.array([0, 0, 0.8]),np.array([0, 0, 0.8]),[0,1,0])
        self.point2b = self.sim.addUserDebugLine(np.array([0, 0, 0.8]),np.array([0, 0, 0.8]),[1,0,0])
        self.line = self.sim.addUserDebugLine(np.array([0, 0, 0.8]),np.array([0, 0, 0.8]),[0,0,1])
        
        os.system('cls')
        self.J_max =np.array([ 90,  80,  55,  90,  90,  150])
        self.J_min =np.array([-90, -30, -50, -90, -90, -150])

        self.state_buffer_size = 1

        self.metadata = {
            'render.modes': ['GUI', 'DIRECT'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._sample_goal_start()
        obs = self._get_obs()
        self.action_space = spaces.Box(-max_Joints_offset, max_Joints_offset, shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,), dtype='float32')
        
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._sim_step(action)
        obs = self._get_obs()
        
        #info = {
        #    'is_success': self._is_success(),
        #}
        reward = self._compute_reward()
        if self.time_step > self.max_step:
            self.done = True
        
        return obs, reward, self.done, self.info
    
    def reset(self):
        self.done = False
        self.info ={'near_collision':False, 'near_limits':False} 
        self.time_step = 0

        self._sample_goal_start()
                
        
        self.state_buffer=[]
        for _ in range (self.state_buffer_size):
            self.state_buffer.extend(self.J_start)

        obs = self._get_obs()
        return obs
    
    def close(self):
        self.sim.disconnect()
        
        
    def _sample_goal_start(self):
        while True:
            J = np.random.rand(6)*(self.J_max-self.J_min)+self.J_min
            #if self.test:
            #    J=np.array([-50.0,0.0,25.0,0.0,20.0,0.0])+np.random.rand(6)*20-10
            collision_distance, joint_distance, xyz, R = self._check_collision(J)
            #if collision_check==0:
            if collision_distance > 5 and joint_distance > 2:
                self.J_start = J.copy()
                self.xyz_start = xyz.copy()
                self.xyz_achieved = xyz.copy()
                self.R_achieved = R.copy()
                self.J_achieved = J.copy()
                break
        while True:
            J = np.random.rand(6)*(self.J_max-self.J_min)+self.J_min
            #if self.test:
            #    J=np.array([ 50.0,0.0,25.0,0.0,20.0,0.0])+np.random.rand(6)*20-10
            #collision_check, xyz, q = self._check_collision(J)
            collision_distance, joint_distance, xyz, q = self._check_collision(J)
            #if collision_check==0 and np.linalg.norm(xyz - self.xyz_start)>100:
            if (collision_distance > 5 and joint_distance > 2) or self.test :
                self.J_goal = J.copy()
                self.xyz_goal = xyz.copy()
                self.R_goal = R.copy()
                break
                
        self.sim.removeUserDebugItem(self.point1a)
        self.sim.removeUserDebugItem(self.point1b)
        self.sim.removeUserDebugItem(self.point2a)
        self.sim.removeUserDebugItem(self.point2b)
        self.sim.removeUserDebugItem(self.line)
        self.point1a = self.sim.addUserDebugLine(self.xyz_start*0.001+np.array([-0.1, 0, 0.8]),self.xyz_start*0.001+np.array([0.1, 0, 0.8]),[0,1,0],3)
        self.point1b = self.sim.addUserDebugLine(self.xyz_start*0.001+np.array([0, 0, 0.8-0.1]),self.xyz_start*0.001+np.array([0, 0, 0.8+0.1]),[0,1,0],3)
        self.point2a = self.sim.addUserDebugLine(self.xyz_goal*0.001+np.array([-0.1, 0, 0.8]),self.xyz_goal*0.001+np.array([0.1, 0, 0.8]),[1,0,0],3)
        self.point2b = self.sim.addUserDebugLine(self.xyz_goal*0.001+np.array([0, 0, 0.8-0.1]),self.xyz_goal*0.001+np.array([0, 0, 0.8+0.1]),[1,0,0],3)
        self.line = self.sim.addUserDebugLine(self.xyz_start*0.001+np.array([0, 0, 0.8]),self.xyz_goal*0.001+np.array([0, 0, 0.8]),[0,0,1],3)

        self.goal_dist = np.linalg.norm(self.xyz_goal-self.xyz_start)
        
    def _sim_step(self,action):
        #print('action:', action)
        self.J_achieved += action
        #H_tool, self.xyz_achieved, R, Q = ABB120.FK(Joints)
        self.collision_distance, self.joint_distance, self.xyz_achieved, self.q_achieved = self._check_collision(self.J_achieved)
        self.time_step += 1

        #for _ in range(3*6):
        for _ in range(6):
            self.state_buffer.pop(0)

        self.state_buffer.extend(self.J_achieved)
        self._compute_reward()
        
        
    def _compute_reward(self):
        reward = 0
        self.done = False
        self.info ={'near_collision':False, 'near_limits':False} 
        if self.collision_distance < 10:
            self.info['near_collision']=True
            reward -= (10-self.collision_distance)*10000000
            #if self.collision_distance < 5:
                #self.done = True
                #print('---------- collision -------------')
                
        if self.joint_distance < 3:
            self.info['near_limits']=True
            reward -= (3-self.joint_distance)*10000000
            #if self.joint_distance < 5:
                #self.done = True
                #print('---------- collision -------------')
                
        #angle, u = angle_axis(self.R_goal*np.transpose(self.R_achieved))
        delta_euler_ZYX = np.linalg.norm(ZYX_from_R(self.R_goal) - ZYX_from_R(self.R_achieved))
        distance = np.linalg.norm(self.xyz_goal - self.xyz_achieved)
        if distance > 30:
            remained = distance + 20*delta_euler_ZYX*180/pi
        else:
            remained = 20*(distance + 50*delta_euler_ZYX*180/pi)
        #print('distance, angle:',np.linalg.norm(self.xyz_goal - self.xyz_achieved), angle*180/pi)
        #remained = np.linalg.norm(self.J_goal - self.J_achieved)
        reward -= remained

        if remained < 10:
            #reward += 100000
            self.done = True
            print("************** Goal achieved! ****************")
        return reward
        
    def _is_success(self):
        remained = np.linalg.norm(self.xyz_goal - self.xyz_achieved)
        if remained < 10:
            print("************** Goal achieved! ****************")
            return True
        else:
            return False
            
    def _check_collision(self,Joints):
        collision_check = 0
        maxDist = 1

        for j in range(6):
            p.resetJointState(self.ABB120Id,j,Joints[j]*math.pi/180)

        dist=[]
        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=-1, bodyB=self.ABB120Id, linkIndexB=1, distance=maxDist)[0][8])
        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=-1, bodyB=self.ABB120Id, linkIndexB=2, distance=maxDist)[0][8])
        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=-1, bodyB=self.ABB120Id, linkIndexB=3, distance=maxDist)[0][8])
        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=-1, bodyB=self.ABB120Id, linkIndexB=4, distance=maxDist)[0][8])
        dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=-1, bodyB=self.ABB120Id, linkIndexB=5, distance=maxDist)[0][8])

        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=0, bodyB=self.ABB120Id, linkIndexB=2, distance=maxDist)[0][8])
        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=0, bodyB=self.ABB120Id, linkIndexB=3, distance=maxDist)[0][8])
        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=0, bodyB=self.ABB120Id, linkIndexB=4, distance=maxDist)[0][8])
        dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=0, bodyB=self.ABB120Id, linkIndexB=5, distance=maxDist)[0][8])

        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=1, bodyB=self.ABB120Id, linkIndexB=3, distance=maxDist)[0][8])
        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=1, bodyB=self.ABB120Id, linkIndexB=4, distance=maxDist)[0][8])
        dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=1, bodyB=self.ABB120Id, linkIndexB=5, distance=maxDist)[0][8])

        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=2, bodyB=self.ABB120Id, linkIndexB=4, distance=maxDist)[0][8])
        dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=2, bodyB=self.ABB120Id, linkIndexB=5, distance=maxDist)[0][8])

        #dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=3, bodyB=self.ABB120Id, linkIndexB=5, distance=maxDist)[0][8])

        dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=2, bodyB=self.tableId, distance=maxDist)[0][8])
        dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=3, bodyB=self.tableId, distance=maxDist)[0][8])
        dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=4, bodyB=self.tableId, distance=maxDist)[0][8])
        dist.append(p.getClosestPoints(bodyA=self.ABB120Id, linkIndexA=5, bodyB=self.tableId, distance=maxDist)[0][8])

        #collision_index=[[-1,1], [-1,2], [-1,3], [-1,2], [-1,5],
        #                [0,2], [0,3], [0,4], [0,5],
        #                [1,3], [1,4], [1,5],
        #                [2, 4], [2,5],
        #                [3, 5],
        #                [-10, 2], [-10, 3], [-10, 4], [-10, 5]]

        #for k in range(0,len(dist)):
        #    if dist[k] < 0.001:
        #        collision_check = 1
        #        dist[k] = 0.001

        H_tool,xyz,R,Q=ABB120.FK(Joints)
        #if xyz[2] < 50:
        #    collision_check = 1

        #for i in range(6):
        #    if Joints[i]>self.J_max[i] or Joints[i]<self.J_min[i]:
        #        collision_check = 1
        joint_dist = []
        for i in range(6):
            joint_dist.append(abs(Joints[i]-self.J_max[i]))
            joint_dist.append(abs(Joints[i]-self.J_min[i]))

        collision_distance = min(dist)
        joint_distance = min(joint_dist)
        
        #collision_distance = 1000
        #joint_distance = 1000

        return collision_distance*1000, joint_distance, xyz, R
        
    
    #def _compute_reward(self, achieved_goal, goal, info):
    #    # Compute distance between goal and the achieved goal.
    #   d = goal_distance(achieved_goal, goal)
    #    if self.reward_type == 'sparse':
    #        return -(d > self.distance_threshold).astype(np.float32)
    #    else:
    #        return -d

    
    def _get_obs(self):
        #print(self.xyz_achieved, self.q_achieved)
        obs = np.concatenate([
            self.J_achieved.copy(),
            self.J_goal.copy(),
        ])

        return obs
    
    def capture(self):
        self.sim.resetDebugVisualizerCamera(  cameraDistance=1, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0.2,0,1])
        img = self.sim.getCameraImage(width= 1600, height= 1200, lightDiffuseCoeff = 1, lightSpecularCoeff = 1, lightDistance = 0.5, shadow = 1, lightDirection=[0,0,3])[2]
        img = np.reshape(np.array(img),(1200,1600,4),order = 'C')
        img = np.delete(img, np.s_[-1:], axis=2)
        img = img*(1./255.)
        return img
