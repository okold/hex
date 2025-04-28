# Copyright 2023
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

"""MCCFR agents trained on games."""

import os
import time
import pickle
import numpy as np
import torch

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python import rl_agent
from open_spiel.python.algorithms import outcome_sampling_mccfr, external_sampling_mccfr


class MCCFRPolicy(policy.Policy):
    """Policy wrapper for MCCFR average policy that handles tensor inputs."""

    def __init__(self, game, player_id, mccfr_policy):
        """Initialize policy wrapper.
        
        Args:
            game: The OpenSpiel game.
            player_id: ID of the player this policy is for.
            mccfr_policy: The underlying MCCFR average policy.
        """
        super().__init__(game, [player_id])
        self._mccfr_policy = mccfr_policy
        self._player_id = player_id
        self._game = game
        self._num_actions = game.num_distinct_actions()

    def action_logits(self, state, player_id=None):
        """Returns unnormalized cumulative strategy logits for a given state."""
        if player_id is None:
            player_id = self._player_id

        # Try to extract information state key
        if isinstance(state, torch.Tensor) or isinstance(state, np.ndarray):
            raise NotImplementedError("Tensor input not supported for raw logits yet.")
        elif hasattr(state, "information_state_string"):
            info_state_key = state.information_state_string(player_id)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

        retrieved_infostate = self._mccfr_policy._infostates.get(info_state_key, None)
        if retrieved_infostate is None:
            # Uniform logits if we don't have information
            return torch.zeros(self._num_actions)

        avg_policy_cumulants = retrieved_infostate[1]  # AVG_POLICY_INDEX=1
        return torch.tensor(avg_policy_cumulants, dtype=torch.float32)


    def action_probabilities(self, state, player_id=None):
        """Returns action probabilities for a state.
        
        Args:
            state: A state object or tensor.
            player_id: Optional player ID.
            
        Returns:
            Dict mapping actions to probabilities.
        """
        if player_id is None:
            player_id = self._player_id
            
        # Handle tensor input (for exploitability calculation)
        if isinstance(state, torch.Tensor) or isinstance(state, np.ndarray):
            # Convert tensor to infostate string using game-specific logic
            if isinstance(state, torch.Tensor):
                infostate_tensor = state.detach().cpu().numpy()
            else:
                infostate_tensor = state
                
            # Use a temporary state to get legal actions
            temp_state = self._game.new_initial_state()
            legal_actions = list(range(self._num_actions))  # Fallback
            
            # Try to recover legal actions from the tensor
            # This is a simplified approach and may need game-specific handling
            probs = np.zeros(self._num_actions)
            
            # Uniform policy as fallback when handling tensor inputs
            probs[legal_actions] = 1.0 / len(legal_actions)
            return {a: probs[a] for a in legal_actions}
            
        # Handle OpenSpiel state object
        elif hasattr(state, "information_state_string"):
            return self._mccfr_policy.action_probabilities(state, player_id)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")


class MCCFRRLAgent(rl_agent.AbstractAgent):
    """RL agent wrapper for MCCFR algorithm."""

    def __init__(self, game, player_id, average_policy):
        """Initialize an MCCFR RL agent wrapper.
        
        Args:
            game: The OpenSpiel game.
            player_id: ID of the player this agent controls.
            average_policy: A policy.Policy object representing the average policy.
        """
        self.player_id = player_id
        self._game = game
        self._policy = MCCFRPolicy(game, player_id, average_policy)
        self._num_actions = game.num_distinct_actions()

    def step(self, time_step, is_evaluation=False):
        """Returns the action to be taken and the policy probabilities.
        
        Args:
            time_step: An instance of rl_environment.TimeStep.
            is_evaluation: Whether this is an evaluation step.
            
        Returns:
            A StepOutput object containing the action and probabilities.
        """
        # Don't act at terminal info states
        if time_step.last():
            return

        info_state = time_step.observations["info_state"][self.player_id]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        
        # If serialized state is available, use it
        if "serialized_state" in time_step.observations:
            state = self._game.deserialize_state(
                time_step.observations["serialized_state"])
            probs = self._policy.action_probabilities(state, self.player_id)
        else:
            # Fallback to uniform policy when state is not available
            probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
        
        # Extract probabilities for legal actions
        action_probs = np.zeros(self._num_actions)
        for action, prob in probs.items():
            if action in legal_actions:
                action_probs[action] = prob
        
        # Normalize probabilities
        sum_probs = action_probs.sum()
        if sum_probs > 0:
            action_probs /= sum_probs
        else:
            # Fallback to uniform random
            action_probs = np.zeros(self._num_actions)
            action_probs[legal_actions] = 1.0 / len(legal_actions)
            
        # Sample an action
        action = np.random.choice(
            legal_actions, 
            p=[action_probs[a] for a in legal_actions] / sum(action_probs[legal_actions])
        )
        
        return rl_agent.StepOutput(action=action, probs={a: action_probs[a] for a in range(self._num_actions)})
    
    def __call__(self, tensor_input):
        """Returns logits (unnormalized scores) instead of probabilities."""
        if len(tensor_input.shape) == 1:
            tensor_input = tensor_input.unsqueeze(0)

        batch_size = tensor_input.shape[0]
        output_logits = []
        
        for i in range(batch_size):
            single_input = tensor_input[i]
            # Instead of getting probabilities, get the unnormalized logits
            # But since the tensor input is problematic, you can just fallback to uniform
            # OR: make the traverser give serialized state and reconstruct properly.

            # Here we assume the tensor contains serialized information to reconstruct state.
            # (this part would need adjustment depending on your OpenSpiel settings)
            logits = torch.zeros(self._num_actions)
            logits.fill_(1.0)  # Placeholder: you could extract more correct info
            output_logits.append(logits)
            
        return torch.stack(output_logits, dim=0)

    def get_model(self):
        """Returns the underlying policy model."""
        return self


class RunMCCFR:
    """Class to manage the training of MCCFR agents."""
    
    def __init__(self, args, game, expl_callback):
        """Initialize the MCCFR runner.
        
        Args:
            args: Configuration arguments.
            game: The OpenSpiel game to train on.
            expl_callback: Callback function for computing exploitability.
        """
        self.args = args
        self.game = game
        self.expl_callback = expl_callback
        self.step_count = 0
        self.mccfr_solver = None
        self.policy = None
        self.avg_policy = None

    def run(self):
        """Run the MCCFR algorithm training process.
        
        Returns:
            Tuple of (path to saved agent, trained solver).
        """
        args = self.args
        ckpt_path = args.experiment_dir
        
        # Extract the algorithm config from args
        algorithm_config = args.algorithm
        
        # Initialize the MCCFR solver
        if algorithm_config.algorithm_type == "outcome":
            self.mccfr_solver = outcome_sampling_mccfr.OutcomeSamplingSolver(self.game)
        elif algorithm_config.algorithm_type == "external":
            self.mccfr_solver = external_sampling_mccfr.ExternalSamplingSolver(self.game)
        else:
            raise ValueError(f"Unknown MCCFR algorithm type: {algorithm_config.algorithm_type}")
        
        # Create the environment for evaluation
        env = rl_environment.Environment(self.game)
        
        # Training loop
        old_time = time.time()
        expl_check_step_count = 0
        iteration = 0
        
        while True:
            # Run iterations of MCCFR
            self.mccfr_solver.iteration()
            self.step_count += 1
            expl_check_step_count += 1
            iteration += 1
            
            # Periodic evaluation
            if iteration % algorithm_config.eval_every == 0:
                cur_time = time.time()
                iteration_time = cur_time - old_time
                steps_per_sec = algorithm_config.eval_every / iteration_time
                
                print(f"Iteration {iteration}, Steps: {self.step_count}, "
                      f"Iterations/sec: {steps_per_sec:.2f}")
                
                old_time = cur_time
            
            # Check exploitability periodically
            if expl_check_step_count >= args.compute_exploitability_every:
                self.avg_policy = self.mccfr_solver.average_policy()
                expl_agents = self.wrap_rl_agent(f"{args.experiment_dir}/agent.pkl")
                
                # Call exploitability callback if provided
                if self.expl_callback is not None:
                    self.expl_callback(expl_agents[0], expl_agents[1], self.step_count)
                
                expl_check_step_count = 0
                
                # Save checkpoint
                self._save_checkpoint(ckpt_path, f"iter_{iteration}")
            
            # Check if we've reached the maximum number of steps
            if self.step_count >= args.max_steps:
                self.avg_policy = self.mccfr_solver.average_policy()
                expl_agents = self.wrap_rl_agent(f"{args.experiment_dir}/agent.pkl")
                
                if self.expl_callback is not None:
                    self.expl_callback(expl_agents[0], expl_agents[1], self.step_count)
                
                break
        
        # Save final checkpoint
        final_path = self._save_checkpoint(ckpt_path, "final")
        return final_path, self.mccfr_solver
    
    def _save_checkpoint(self, ckpt_path, tag):
        """Save a checkpoint of the MCCFR solver and policy.
        
        Args:
            ckpt_path: Directory to save checkpoint.
            tag: Tag to add to checkpoint filename.
            
        Returns:
            Path to saved checkpoint.
        """
        os.makedirs(ckpt_path, exist_ok=True)
        
        # Save infostates
        infostates_path = os.path.join(ckpt_path, f"infostates_{tag}.pkl")
        with open(infostates_path, "wb") as f:
            pickle.dump(self.mccfr_solver._infostates, f)
        
        # Save agent
        agent_path = os.path.join(ckpt_path, f"agent_{tag}.pkl")
        self.wrap_rl_agent(agent_path)
        
        return ckpt_path
    
    def current_step(self):
        """Returns the current step count."""
        return self.step_count
    
    def wrap_rl_agent(self, save_path=None):
        """Wrap the MCCFR policy as an RL agent.
        
        Args:
            save_path: Path to save the wrapped agent.
            
        Returns:
            List of MCCFR RL agents, one per player.
        """
        if self.avg_policy is None:
            self.avg_policy = self.mccfr_solver.average_policy()
        
        agents = [
            MCCFRRLAgent(self.game, player_id, self.avg_policy)
            for player_id in range(self.game.num_players())
        ]
        
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(agents, f)
        
        return agents
