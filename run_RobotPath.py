#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import logger
import RobotPath


def train(num_timesteps, seed):
    import mlp_policy, pposgd_simple
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 10000 * rank
    set_global_seeds(workerseed)
    env = RobotPath.env(False)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=512, num_hid_layers=3)
    
    
    pposgd_simple.learn(sess, env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=256,
            gamma=0.99, lam=0.95, schedule='linear'
        )
    env.close()


def main():
    train(num_timesteps=1e8, seed=0)


if __name__ == '__main__':
    main()
