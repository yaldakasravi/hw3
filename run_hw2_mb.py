import os
import time

import sys
print(sys.path)


from hw2.roble.agents.mb_agent import MBAgent
from hw2.roble.infrastructure.rl_trainer import RL_Trainer
import hydra, json
from omegaconf import DictConfig, OmegaConf

class MB_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'ensemble_size': params['alg']['ensemble_size'],
            'n_layers': params['alg']['n_layers'],
            'size': params['alg']['size'],
            'learning_rate': params['alg']['learning_rate'],
            }

        train_args = {
            'num_agent_train_steps_per_iter': params['alg']['num_agent_train_steps_per_iter'],
            'discrete': False,
            'ob_dim':  0,
            'ac_dim': 0,
        }

        controller_args = {
            'mpc_horizon': params['alg']['mpc_horizon'],
            'mpc_num_action_sequences': params['alg']['mpc_num_action_sequences'],
            'mpc_action_sampling_strategy': params['alg']['mpc_action_sampling_strategy'],
            'cem_iterations': params['alg']['cem_iterations'],
            'cem_num_elites': params['alg']['cem_num_elites'],
            'cem_alpha': params['alg']['cem_alpha'],
        }

        agent_params = {**computation_graph_args, **train_args, **controller_args}

        tmp = OmegaConf.create({'agent_params' : agent_params })

        self.params = OmegaConf.merge(tmp , params)
        print(self.params)

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params , agent_class =  MBAgent)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['alg']['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )


@hydra.main(config_path="conf", config_name="config_hw2")
def my_main(cfg: DictConfig):
    my_app(cfg)


def my_app(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))
    import os
    print("Command Dir:", os.getcwd())
    # print ("params: ", json.dumps(params, indent=4))
    if cfg['env']['env_name']=='reacher-roble-v0':
        cfg['env']['max_episode_length']=200
    if cfg['env']['env_name']=='cheetah-roble-v0':
        cfg['env']['max_episode_length']=500
    if cfg['env']['env_name']=='obstacles-roble-v0':
        cfg['env']['max_episode_length']=100
    params = vars(cfg)
    print ("params: ", params)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################


    logdir_prefix = 'hw2_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    exp_name = logdir_prefix + cfg.env.exp_name + '_' + cfg.env.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, exp_name)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.logging.logdir = logdir
        cfg.logging.exp_name = exp_name

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = MB_Trainer(cfg)
    trainer.run_training_loop()



if __name__ == "__main__":
    import os
    print("Command Dir:", os.getcwd())
    my_main()