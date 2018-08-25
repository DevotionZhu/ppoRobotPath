#!/usr/bin/env python3

from mpi4py import MPI
from baselines import logger
import RobotPath
import tensorflow as tf
import time

num_episodes = 100

def run():
    import mlp_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    env = RobotPath.env( render=True, max_step=2048)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=512, num_hid_layers=3)
    
    pi = policy_fn('pi', env.observation_space, env.action_space)
    #tf.train.Saver().restore(sess, './model/RobotPath_model')
    U.load_state('./model/RobotPath_model')
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            #start_time = time.time()
            action = pi.act(False, obs)[0]
            #end_time = time.time()
            #print("elapsed time:",end_time-start_time)
            obs, reward, done, info = env.step(action*0.1)
        


def main():
    run()


if __name__ == '__main__':
    main()
