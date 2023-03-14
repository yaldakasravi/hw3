import abc
import itertools
import numpy as np
import torch
from hw3.roble.util import class_util as classu

from hw3.roble.infrastructure import pytorch_util as ptu
from hw3.roble.policies.base_policy import BasePolicy
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    @classu.hidden_member_initialize
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 deterministic=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars

        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           n_layers=self._n_layers,
                                           size=self._size)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                        self._learning_rate)
        else:
            self._logits_na = None
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim,
                                      output_size=self._ac_dim,
                                      n_layers=self._n_layers, size=self._size)
            self._mean_net.to(ptu.device)
            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._logstd = nn.Parameter(
                    torch.zeros(self._ac_dim, dtype=torch.float32, device=ptu.device)
                )
                self._logstd.to(ptu.device)
                self._optimizer = optim.Adam(
                    itertools.chain([self._logstd], self._mean_net.parameters()),
                    self._learning_rate
                )

        if nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                n_layers=self._n_layers,
                size=self._size,
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation_tensor = torch.tensor(observation, dtype=torch.float).to(ptu.device)
        action_distribution = self.forward(observation_tensor)
        return cast(
            np.ndarray,
            action_distribution.sample().cpu().detach().numpy(),
        )
        # TODO: 
        ##


    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            if self._deterministic:
                ##  TODO output for a deterministic policy
                action_distribution = distributions.Normal(
                self.mean_net(observation),
                torch.exp(self.logstd)[None],
                )
                return action_distribution
            else:
                batch_mean = self._mean_net(observation)
                scale_tril = torch.diag(torch.exp(self._logstd))
                batch_dim = batch_mean.shape[0]
                batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
                action_distribution = distributions.MultivariateNormal(
                    batch_mean,
                    scale_tril=batch_scale_tril,
                )
        return action_distribution

class ConcatMLP(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self._dim)
        return super().forward(flat_inputs, **kwargs)

class MLPPolicyDeterministic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, **kwargs):
        kwargs['deterministic'] = True
        super().__init__(*args, **kwargs)
        
    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss

        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32)

        # Compute the action probabilities based on the current policy
        action_probs = self.policy(observations)

        # Compute the Q-values for the current state-action pairs using the Q-function
        q_values = q_fun(observations)

        # Compute the loss as the negative log-likelihood of the selected actions multiplied by the Q-values
        log_probs = torch.log(action_probs)
        selected_log_probs = torch.sum(log_probs * self.selected_actions, dim=1)
        loss = torch.mean(selected_log_probs * q_values)

        # Zero out the gradients of the policy optimizer
        self.policy_optimizer.zero_grad()

        # Compute the gradients of the loss with respect to the policy parameters
        loss.backward()

        # Update the policy parameters using the gradients
        self.policy_optimizer.step()

        # Return the loss as a scalar value
        return loss.item()
    
class MLPPolicyStochastic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, entropy_coeff, *args, **kwargs):
        kwargs['deterministic'] = False
        super().__init__(*args, **kwargs)
        self.entropy_coeff = entropy_coeff

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: sample actions from the gaussian distribrution given by MLPPolicy policy when providing the observations.
        # Hint: make sure to use the reparameterization trick to sample from the distribution
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

            # Get the mean and standard deviation of the action distribution from the policy network
        mean, log_std = self.policy(obs)

        # Sample a normal distribution with the given mean and standard deviation using the reparameterization trick
        normal = torch.distributions.Normal(mean, torch.exp(log_std))
        action = normal.rsample()

        # Clamp the action to be within the action bounds
        action = torch.clamp(action, self.action_low, self.action_high)
        return ptu.to_numpy(action)
        
    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        ## Hint: you will have to add the entropy term to the loss using self.entropy_coeff
        # Convert observations to tensor if necessary
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32)

        # Get the mean and log standard deviation of the action distribution from the policy network
        mean, log_std = self.policy(observations)

        # Sample an action from the policy network
        normal = torch.distributions.Normal(mean, torch.exp(log_std))
        action = normal.rsample()

        # Compute the log probabilities of the sampled action
        log_prob = normal.log_prob(action)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)

        # Compute the entropy of the action distribution
        entropy = normal.entropy().mean()

        # Compute the Q-values for the current state-action pairs using the Q-function
        q_values = q_fun(observations, action)

        # Compute the loss as the sum of the negative log probabilities of the actions multiplied by the Q-values
        loss = -torch.mean(log_prob * q_values) - self.entropy_coeff * entropy

        # Zero out the gradients of the policy optimizer
        self.policy_optimizer.zero_grad()

        # Compute the gradients of the loss with respect to the policy parameters
        loss.backward()

        # Update the policy parameters using the gradients
        self.policy_optimizer.step()

        # Return the loss as a scalar value
        return loss.item()
    
#####################################################