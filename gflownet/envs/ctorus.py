"""
Classes to represent hyper-torus environments
"""
import itertools
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.distributions import Categorical, MixtureSameFamily, Uniform, VonMises
from torchtyping import TensorType

from gflownet.envs.htorus import HybridTorus


class ContinuousTorus(HybridTorus):
    """
    Purely continuous (no discrete actions) hyper-torus environment in which the
    action space consists of the increment Delta theta of the angle at each dimension.
    The trajectory is of fixed length length_traj.

    The states space is the concatenation of the angle (in radians and within [0, 2 *
    pi]) at each dimension and the number of actions.

    Attributes
    ----------
    ndim : int
        Dimensionality of the torus

    length_traj : int
       Fixed length of the trajectory.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        The actions are tuples of length n_dim, where the value at position d indicates
        the increment of dimension d. EOS is indicated by increments of np.inf for all
        dimensions.

        This method defines self.eos and the returned action space is
        simply a representative (arbitrary) action with an increment of 0.0 in all
        dimensions.
        """
        self.eos = tuple([np.inf] * self.n_dim)
        self.representative_action = tuple([0.0] * self.n_dim)
        return [self.representative_action]

    def get_policy_output(self, params: dict):
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension d of the hyper-torus and component c of the mixture, the
        output of the policy should return 1) the weight of the component in the
        mixture, 2) the location of the von Mises distribution and 3) the concentration
        of the von Mises distribution to sample the increment of the angle.

        Therefore, the output of the policy model has dimensionality D x C x 1, where D
        is the number of dimensions (self.n_dim) and C is the number of components
        (self.n_comp). In sum, the entries of the entries of the policy output are:

        - d * c * 3 + 0: weight of component c in the mixture for dim. d
        - d * c * 3 + 1: location of Von Mises distribution for dim. d, comp. c
        - d * c * 3 + 2: log concentration of Von Mises distribution for dim. d, comp. c
        """
        policy_output = np.ones(self.n_dim * self.n_comp * 3)
        policy_output[1::3] = params["vonmises_mean"]
        policy_output[2::3] = params["vonmises_concentration"]
        return policy_output

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns [True] if the only possible action is eos, [False] otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True]
        elif state[-1] >= self.length_traj:
            return [True]
        else:
            return [False]

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        Returns [True] if the only possible action is returning to source, [False]
        otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True]
        elif state[-1] == 1:
            return [True]
        else:
            return [False]

    def get_parents(
        self, state: List = None, done: bool = None, action: Tuple[int, float] = None
    ) -> Tuple[List[List], List[Tuple[int, float]]]:
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length n_angles where each element
            is the position at each dimension.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : int
            Last action performed

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        # If source state
        elif state[-1] == 0:
            return [], []
        else:
            for dim, angle in enumerate(action):
                state[int(dim)] = (state[int(dim)] - angle) % (2 * np.pi)
            state[-1] -= 1
            parents = [state]
            return parents, [action]

    def action2representative(self, action: Tuple) -> Tuple:
        """
        Returns the arbirary, representative action in the action space, so that the
        action can be contrasted with the action space and masks.
        """
        return self.representative_action

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: Optional[TensorType["n_states", "policy_input_dim"]] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs. The angle increments
        that form the actions are sampled from a mixture of Von Mises distributions.

        A distinction between forward and backward actions is made and specified by the
        argument is_backward, in order to account for the following special cases:

        Forward:

        - If the number of steps is equal to the maximum, then the only valid action is
          EOS.

        Backward:

        - If the number of steps is equal to 1, then the only valid action is to return
          to the source. The specific action depends on the current state.

        Args
        ----
        policy_outputs : tensor
            The output of the GFlowNet policy model.

        mask : tensor
            The mask containing information about special cases.

        states_from : tensor
            The states originating the actions, in policy format.

        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default). 
        """
        device = policy_outputs.device
        mask_states_sample = ~mask_invalid_actions.flatten()
        n_states = policy_outputs.shape[0]
        # Sample angle increments
        angles = torch.zeros(n_states, self.n_dim).to(device)
        logprobs = torch.zeros(n_states, self.n_dim).to(device)
        if torch.any(mask_states_sample):
            if sampling_method == "uniform":
                distr_angles = Uniform(
                    torch.zeros(len(ns_range_noeos)),
                    2 * torch.pi * torch.ones(len(ns_range_noeos)),
                )
            elif sampling_method == "policy":
                mix_logits = policy_outputs[mask_states_sample, 0::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                mix = Categorical(logits=mix_logits)
                locations = policy_outputs[mask_states_sample, 1::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                concentrations = policy_outputs[mask_states_sample, 2::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                vonmises = VonMises(
                    locations,
                    torch.exp(concentrations) + self.vonmises_min_concentration,
                )
                distr_angles = MixtureSameFamily(mix, vonmises)
            angles[mask_states_sample] = distr_angles.sample()
            logprobs[mask_states_sample] = distr_angles.log_prob(
                angles[mask_states_sample]
            )
        logprobs = torch.sum(logprobs, axis=1)
        # Build actions
        # TODO: the line below assumes forward sampling only (EOS action if mask is
        # True) - if mask_states_sample is False.
        # TODO: we need to know the states in order to sample the back to source
        # backward action.
        actions_tensor = torch.inf * torch.ones(
            angles.shape, dtype=self.float, device=device
        )
        actions_tensor[mask_states_sample, :] = angles[mask_states_sample]
        actions = [tuple(a.tolist()) for a in actions_tensor]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        is_forward: bool,
        actions: TensorType["n_states", "n_dim"],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask_invalid_actions: TensorType["n_states", "1"] = None,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.
        """
        device = policy_outputs.device
        mask_states_sample = ~mask_invalid_actions.flatten()
        n_states = policy_outputs.shape[0]
        logprobs = torch.zeros(n_states, self.n_dim).to(device)
        if torch.any(mask_states_sample):
            mix_logits = policy_outputs[mask_states_sample, 0::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            mix = Categorical(logits=mix_logits)
            locations = policy_outputs[mask_states_sample, 1::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            concentrations = policy_outputs[mask_states_sample, 2::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            vonmises = VonMises(
                locations,
                torch.exp(concentrations) + self.vonmises_min_concentration,
            )
            distr_angles = MixtureSameFamily(mix, vonmises)
            logprobs[mask_states_sample] = distr_angles.log_prob(
                actions[mask_states_sample]
            )
        logprobs = torch.sum(logprobs, axis=1)
        return logprobs

    def _step(
        self,
        action: Tuple[float],
        backward: bool,
        skip_mask_check: bool = False,
    ) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes step given an action. This method is called by both step() and
        step_backwards(), with the corresponding value of argument backward.

        Forward steps:
            - Add action increments to state angles.
            - Increment n_actions value of state.
        Backward steps:
            - Subtract action increments from state angles.
            - Decrement n_actions value of state.

        Args
        ----
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        backward : bool
            If True, perform backward step. Otherwise (default), perform forward step.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        self.n_actions += 1
        for dim, angle in enumerate(action):
            if backward:
                self.state[int(dim)] -= angle
            else:
                self.state[int(dim)] += angle
            self.state[int(dim)] = self.state[int(dim)] % (2 * np.pi)
        if backward:
            self.state[-1] -= 1
        else:
            self.state[-1] += 1
        return self.state, action, True

    def step(
        self, action: Tuple[float], skip_mask_check: bool = False
    ) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes forward step given an action.

        See: _step().

        Args
        ----
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        skip_mask_check : bool
            Ignored because the action space space is fully continuous, therefore there
            is nothing to check.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if self.done:
            return self.state, action, False
        # If only possible action is EOS (number of actions is equal to maximum
        # trajectory length), then force EOS.
        elif self.n_actions == self.length_traj:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If anyway action is EOS, then it is invalid
        elif action == self.eos:
            # [DEBUG] this point should never be reached. Consider removing this elif.
            print(
                "An EOS action has been sampled forward even though it is not allowed"
            )
            breakpoint()
            return self.state, action, False
        # Otherwise perform action
        else:
            return self._step(action, backward=False)

    def step_backwards(
        self, action: Tuple[float], skip_mask_check: bool = False
    ) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes backward step given an action.

        See: _step().

        Args
        ----
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        skip_mask_check : bool
            Ignored because the action space space is fully continuous, therefore there
            is nothing to check.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # If self.done is True, set done to False, increment n_actions and return same
        # state
        if self.done:
            assert action == self.eos
            self.done = False
            self.n_actions += 1
            return self.state, action, True
        # If the number of actions is equal to the maximum, step should not proceed.
        elif self.n_actions == self.length_traj:
            return self.state, self.eos, False
        # If anyway action is EOS, then it is invalid
        elif action == self.eos:
            # [DEBUG] this point should never be reached. Consider removing this elif.
            print(
                "An EOS action has been sampled backward even though it is not allowed"
            )
            breakpoint()
            return self.state, action, False
        # Otherwise perform action
        else:
            return self._step(action, backward=True)
