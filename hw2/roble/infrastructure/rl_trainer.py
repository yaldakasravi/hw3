from collections import OrderedDict
from gym import wrappers

# hw1 imports
from hw1.roble.infrastructure.rl_trainer import RL_Trainer
from hw1.roble.infrastructure import pytorch_util as ptu

# hw2 imports
from hw2.roble.agents.mb_agent import MBAgent
from hw2.roble.infrastructure import utils
# register all of our envs
from hw2.roble.envs import register_envs

import gym
import numpy as np
import pickle
import os
import sys
import time
import torch

register_envs()

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 1
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

class RL_Trainer(RL_Trainer):

    def __init__(self, params , agent_class):

        # Inherit from hw1 RL_Trainer
        super().__init__(params, agent_class)

        # Make the gym environment
        self.create_env(self.params['env']['env_name'])
        print (self.env) 
        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.Monitor(self.env, os.path.join(self.params['logging']['logdir'], "gym"), force=True)
            self.env = params['env_wrappers'](self.env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        if 'non_atari_colab_env' in self.params and self.params['logging']['video_log_freq'] > 0:
            self.env = wrappers.Monitor(self.env, os.path.join(self.params['logging']['logdir'], "gym"), force=True)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')

        # import plotting (locally if 'obstacles' env)
        if not(self.params['env']['env_name']=='obstacles-roble-v0'):
            import matplotlib
            matplotlib.use("Agg")

        # Maximum length for episodes
        self.params['env']['max_episode_length'] = self.params['env']['max_episode_length'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['env']['max_episode_length']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10


        #############
        ## AGENT
        #############

        self.agent = agent_class(self.env, self.params['agent_params'])
        
    def create_env(self, env_name):
        self.env = gym.make(env_name)

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.total_train_rewards = 0
        self.total_eval_rewards = 0
        self.start_time = time.time()

        print_period = 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['logging']['video_log_freq'] == 0 and self.params['logging']['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if self.params['logging']['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['logging']['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            use_batchsize = self.params['alg']['batch_size']
            if itr == 0:
                use_batchsize = self.params['alg']['batch_size_initial']
            paths, envsteps_this_batch, train_video_paths = (
                self.collect_training_trajectories(
                    itr, initial_expertdata, collect_policy, use_batchsize)
            )

            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            if isinstance(self.agent, MBAgent):
                self.agent.add_to_replay_buffer(paths, self.params['alg']['add_sl_noise'])
            else:
                self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()

            # if there is a model, log model predictions
            if isinstance(self.agent, MBAgent) and itr == 0:
                self.log_model_predictions(itr, all_logs)

            # log/save
            if self.log_video or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self.params['logging']['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logging']['logdir'], itr))

    ####################################
    ####################################

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['alg']['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['alg']['train_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################
    

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['alg']['eval_batch_size'], self.params['env']['max_episode_length'])

        # save eval rollouts as videos in the video folder (for grading)
        if self.log_video:
            if train_video_paths is not None:
                #save train/eval videos
                print('\nSaving train rollouts as videos...')
                self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
            print('\nSaving eval rollouts as videos...')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')
        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["Train_AverageRewardSoFar"] = self.total_train_rewards / self.total_envsteps
            logs["Eval_AverageRewardSoFar"] = self.total_eval_rewards / self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_file(value, key,itr)
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

    def log_model_predictions(self, itr, all_logs):
        # model predictions
        import matplotlib.pyplot as plt
        self.fig = plt.figure()

        # sample actions
        action_sequence = self.agent.actor.sample_action_sequences(num_sequences=1, horizon=10) #20 reacher
        action_sequence = action_sequence[0]

        # calculate and log model prediction error
        mpe, true_states, pred_states = utils.calculate_mean_prediction_error(self.env, action_sequence, self.agent.dyn_models, self.agent.actor.data_statistics)
        assert self.params['agent_params']['ob_dim'] == true_states.shape[1] == pred_states.shape[1]
        ob_dim = self.params['agent_params']['ob_dim']
        ob_dim = 2*int(ob_dim/2.0) ## skip last state for plotting when state dim is odd

        # plot the predictions
        self.fig.clf()
        for i in range(ob_dim):
            plt.subplot(ob_dim/2, 2, i+1)
            plt.plot(true_states[:,i], 'g')
            plt.plot(pred_states[:,i], 'r')
        self.fig.suptitle('MPE: ' + str(mpe))
        self.fig.savefig(self.params['logging']['logdir']+'/itr_'+str(itr)+'_predictions.png', dpi=200, bbox_inches='tight')

        # plot all intermediate losses during this iteration
        all_losses = np.array([log['Training Loss'] for log in all_logs])
        np.save(self.params['logging']['logdir']+'/itr_'+str(itr)+'_losses.npy', all_losses)
        self.fig.clf()
        plt.plot(all_losses)
        self.fig.savefig(self.params['logging']['logdir']+'/itr_'+str(itr)+'_losses.png', dpi=200, bbox_inches='tight')

