# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural Fictitious Self-Play (NFSP) agent implemented in TensorFlow.

See the paper https://arxiv.org/abs/1603.01121 for more details.
"""

import collections
import contextlib
import enum
import os
import random
import numpy as np

# import tensorflow.compat.v1 as tf
import torch
from open_spiel.python import rl_agent


# from algorithms.dqn import dqn
from open_spiel.python.pytorch import dqn
from algorithms.ppo.ppo_wrapper import PPOWrapper
from algorithms.ppo.ppo import PPOAgent

Transition = collections.namedtuple(
    "Transition", "info_state action_probs legal_actions_mask"
)

MODE = enum.Enum("mode", "best_response average_policy")


class NFSP(rl_agent.AbstractAgent):
    """NFSP Agent implementation in TensorFlow.

    See open_spiel/python/examples/kuhn_nfsp.py for an usage example.
    """

    def __init__(
        self,
        player_id,
        state_representation_size,
        num_actions,
        hidden_layers_sizes,
        reservoir_buffer_capacity=int(2e6),
        anticipatory_param=0.1,
        batch_size=128,
        rl_learning_rate=0.01,
        sl_learning_rate=0.01,
        min_buffer_size_to_learn=1000,
        learn_every=64,
        optimizer_str="sgd",
        inner_rl_agent_type="dqn",
        inner_rl_agent_args=None,
        **kwargs,
    ):
        """Initialize the `NFSP` agent."""
        self.player_id = player_id
        # self._session = session
        self._num_actions = num_actions
        self._layer_sizes = hidden_layers_sizes
        self._batch_size = batch_size
        self._learn_every = learn_every
        self._anticipatory_param = anticipatory_param
        self._min_buffer_size_to_learn = min_buffer_size_to_learn

        self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)
        self._prev_timestep = None
        self._prev_action = None
        # Step counter to keep track of learning.
        self._step_counter = 0
        self._br_steps = 0
        self.max_steps = kwargs.get("max_steps", 0)
        # Inner RL agent
        self._inner_rl_agent_type = inner_rl_agent_type
        if self._inner_rl_agent_type == "dqn":
            dqn_args = {
                k: v for k, v in inner_rl_agent_args.items() if k != "algorithm_name"
            }
            self._rl_agent = dqn.DQN(
                player_id,
                state_representation_size,
                num_actions,
                hidden_layers_sizes,
                **dqn_args,
            )
        elif self._inner_rl_agent_type == "ppo":
            ppo_hparams = {
                "use_wandb": False,
                "agent_fn": PPOAgent,
                **inner_rl_agent_args,
            }
            self._rl_agent = PPOWrapper(
                player_id, state_representation_size, num_actions, **ppo_hparams
            )
            self._rl_agent.anneal_lr = True
        else:
            raise ValueError("Not implemented. Choose from ['dqn', 'ppo'].")

        # Keep track of the last training loss achieved in an update step.
        if self._inner_rl_agent_type == "dqn":
            self._last_rl_loss_value = lambda: self._rl_agent.loss
        else:
            self._last_rl_loss_value = lambda: 0
            self._last_sl_loss_value = None

        # # Placeholders.
        self._info_state_ph = torch.zeros(
            (1, state_representation_size)
        )  # .to(torch.double)# tf.placeholder(

        # Average policy network.
        self._avg_network = dqn.MLP(
            state_representation_size, self._layer_sizes, num_actions
        )
        self._avg_policy = self._avg_network(self._info_state_ph.to(torch.float32))
        self._avg_policy_probs = torch.softmax(self._avg_policy, dim=1)

        # Loss
        self._loss = torch.nn.CrossEntropyLoss()

        if optimizer_str == "adam":
            self.optimizer = torch.optim.Adam(
                list(self._avg_network.parameters()), lr=sl_learning_rate
            )
        elif optimizer_str == "sgd":
            self.optimizer = torch.optim.SGD(
                self._avg_network.parameters(), lr=sl_learning_rate
            )
        else:
            raise ValueError("Not implemented. Choose from ['adam', 'sgd'].")

        self._sample_episode_policy()

    @contextlib.contextmanager
    def temp_mode_as(self, mode):
        """Context manager to temporarily overwrite the mode."""
        previous_mode = self._mode
        self._mode = mode
        yield
        self._mode = previous_mode

    def get_step_counter(self):
        return self._step_counter

    def _sample_episode_policy(self):
        if np.random.rand() < self._anticipatory_param:
            self._mode = MODE.best_response
        else:
            self._mode = MODE.average_policy

    def _act(self, info_state, legal_actions):
        info_state = np.reshape(info_state, [1, -1])

        action_values = self._avg_network(torch.as_tensor(info_state).to(torch.float32))

        legal_action_values = torch.zeros(1, self._num_actions) - 1e6
        legal_action_values[0, legal_actions] = action_values[0][legal_actions].detach()

        # this is not necessary because torch softmax is already stable but why not be safe.
        legal_action_values -= torch.max(legal_action_values, dim=-1)[0]

        action_probs = torch.softmax(legal_action_values, dim=1)

        self._last_action_values = action_values[0]

        # Remove illegal actions, normalize probs
        probs = np.zeros(self._num_actions)
        probs[legal_actions] = action_probs[0][legal_actions].detach().numpy()
        probs /= sum(probs)

        action = np.random.choice(len(probs), p=probs)

        return action, probs

    @property
    def mode(self):
        return self._mode

    @property
    def loss(self):
        return (self._last_sl_loss_value, self._last_rl_loss_value())

    def step(self, time_step, is_evaluation=False):
        """Returns the action to be taken and updates the Q-networks if needed.

        Args:
        time_step: an instance of rl_environment.TimeStep.
        is_evaluation: bool, whether this is a training or evaluation call.

        Returns:
        A `rl_agent.StepOutput` containing the action probs and chosen action.
        """

        # Prepare for the next episode.
        # for the ppo agent, we have to sample the episode policy at the beginning of the episode because of the post episode step:
        # if time_step.first() and self._inner_rl_agent_type == "ppo":
        #   self._sample_episode_policy()

        if self._mode == MODE.best_response:
            agent_output = self._rl_agent.step(time_step, is_evaluation)
            if not is_evaluation and not time_step.last():
                self._add_transition(time_step, agent_output)

        elif self._mode == MODE.average_policy:
            # Act step: don't act at terminal info states.
            if not time_step.last():
                info_state = time_step.observations["info_state"][self.player_id]
                legal_actions = time_step.observations["legal_actions"][self.player_id]
                action, probs = self._act(info_state, legal_actions)
                agent_output = rl_agent.StepOutput(action=action, probs=probs)

            if self._prev_timestep and not is_evaluation:
                self._rl_agent.add_transition(
                    self._prev_timestep, self._prev_action, time_step
                )
        else:
            raise ValueError("Invalid mode ({})".format(self._mode))

        if not is_evaluation:
            self._step_counter += 1

            if self._step_counter % self._learn_every == 0:
                self._last_sl_loss_value = self._learn()
                if (
                    self._mode == MODE.average_policy
                    and self._inner_rl_agent_type == "dqn"
                ):
                    self._rl_agent.learn()

            if time_step.last():
                self._sample_episode_policy()
                self._prev_timestep = None
                self._prev_action = None
                return
            else:
                self._prev_timestep = time_step
                self._prev_action = agent_output.action

        return agent_output

    def _add_transition(self, time_step, agent_output):
        """Adds the new transition using `time_step` to the reservoir buffer.

        Transitions are in the form (time_step, agent_output.probs, legal_mask).

        Args:
          time_step: an instance of rl_environment.TimeStep.
          agent_output: an instance of rl_agent.StepOutput.
        """
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        transition = Transition(
            info_state=(time_step.observations["info_state"][self.player_id][:]),
            action_probs=agent_output.probs,
            legal_actions_mask=legal_actions_mask,
        )
        self._reservoir_buffer.add(transition)

    def _learn(self):
        """Compute the loss on sampled transitions and perform a avg-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Returns:
          The average loss obtained on this batch of transitions or `None`.
        """
        if (
            len(self._reservoir_buffer) < self._batch_size
            or len(self._reservoir_buffer) < self._min_buffer_size_to_learn
        ):
            return None

        transitions = self._reservoir_buffer.sample(self._batch_size)
        info_states = [t.info_state for t in transitions]
        action_probs = [t.action_probs for t in transitions]

        info_states = torch.as_tensor(np.stack(info_states)).to(torch.float32)
        action_probs = torch.as_tensor(np.stack(action_probs)).to(torch.float32)

        logits = self._avg_network(info_states)
        loss = self._loss(logits, action_probs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _full_checkpoint_name(self, checkpoint_dir, name, episode):
        checkpoint_filename = (
            "_".join([name, "pid" + str(self.player_id), "ep" + str(episode)]) + ".pt"
        )
        return os.path.join(checkpoint_dir, checkpoint_filename)

    def _latest_checkpoint_filename(self, name):
        checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
        return checkpoint_filename + "_latest"

    def save(self, checkpoint_dir, episode):
        """Saves the average policy network and the inner RL agent's q-network.

        Note that this does not save the experience replay buffers and should
        only be used to restore the agent's policy, not resume training.

        Args:
          checkpoint_dir: directory where checkpoints will be saved.
        """
        torch.save(
            self._avg_network.state_dict(),
            self._full_checkpoint_name(checkpoint_dir, "avg_network", episode),
        )
        if self._inner_rl_agent_type == "dqn":
            torch.save(
                self._rl_agent._q_network.state_dict(),
                self._full_checkpoint_name(checkpoint_dir, "q_network", episode),
            )
        return self._full_checkpoint_name(checkpoint_dir, "q_network", episode)

    def has_checkpoint(self, checkpoint_dir):
        pass

    def restore(self, checkpoint_dir):
        """Restores the average policy network and the inner RL agent's q-network.

        Note that this does not restore the experience replay buffers and should
        only be used to restore the agent's policy, not resume training.

        Args:
          checkpoint_dir: directory from which checkpoints will be restored.
        """
        # for name, saver in self._savers:
        #   full_checkpoint_dir = self._full_checkpoint_name(checkpoint_dir, name)
        #   logging.info("Restoring checkpoint: %s", full_checkpoint_dir)
        #   saver.restore(self._session, full_checkpoint_dir)
        pass


class ReservoirBuffer(object):
    """Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    """

    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        """Potentially adds `element` to the reservoir buffer.

        Args:
          element: data to be added to the reservoir buffer.
        """
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        """Returns `num_samples` uniformly sampled from the buffer.

        Args:
          num_samples: `int`, number of samples to draw.

        Returns:
          An iterable over `num_samples` random elements of the buffer.

        Raises:
          ValueError: If there are less than `num_samples` elements in the buffer
        """
        if len(self._data) < num_samples:
            raise ValueError(
                "{} elements could not be sampled from size {}".format(
                    num_samples, len(self._data)
                )
            )
        return random.sample(self._data, num_samples)

    def clear(self):
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)
