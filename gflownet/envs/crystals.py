"""
Classes to represent crystal environments
"""
import itertools
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from gflownet.envs.base import GFlowNetEnv


class Crystal(GFlowNetEnv):
    """
    Crystal environment for ionic conductivity

    Attributes
    ----------
    max_diff_elem : int
        Maximum number of unique elements in the crystal

    min_diff_elem : int
        Minimum number of unique elements in the crystal

    periodic_table : int
        Total number of unique elements that can be used for building the crystal

    min_atoms : int
        Minimum number of atoms that needs to be used to construct a crystal

    max_atoms : int
        Maximum number of atoms that can be used to construct a crystal

    min_atom_i : int
        Minimum number of elements of each used kind that needs to be used to construct a crystal

    max_atom_i : int
        Maximum number of elements of each kind that can be used to construct a crystal

    oxidation_states : (optional) dict
        Mapping from ints (representing elements) to lists of different oxidation states

    alphabet : (optional) dict
        Mapping from ints (representing elements) to strings containing human-readable elements' names

    required_elements : (optional) list
        List of elements that must be present in a crystal for it to represent a valid end state
    """

    def __init__(
        self,
        max_diff_elem: int = 4,
        min_diff_elem: int = 2,
        periodic_table: int = 84,
        min_atoms: int = 2,
        max_atoms: int = 20,
        min_atom_i: int = 1,
        max_atom_i: int = 10,
        oxidation_states: Optional[Dict] = None,
        alphabet: Optional[Dict] = None,
        required_elements: Optional[List] = None,
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        reward_norm_std_mult=0,
        reward_func="power",
        denorm_proxy=False,
        energies_stats=None,
        proxy=None,
        oracle=None,
        **kwargs,
    ):
        super(Crystal, self).__init__(
            env_id,
            reward_beta,
            reward_norm,
            reward_norm_std_mult,
            reward_func,
            energies_stats,
            denorm_proxy,
            proxy,
            oracle,
            **kwargs,
        )

        self.state = [0 for _ in range(periodic_table)]  # atom counts for each element
        self.max_diff_elem = max_diff_elem
        self.min_diff_elem = min_diff_elem
        self.periodic_table = periodic_table
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.min_atom_i = min_atom_i
        self.max_atom_i = max_atom_i
        self.oxidation_states = oxidation_states
        self.alphabet = alphabet if alphabet is not None else {}
        self.required_elements = (
            required_elements if required_elements is not None else []
        )
        self.obs_dim = self.periodic_table
        self.action_space = self.get_actions_space()
        self.eos = len(self.action_space)

    def get_actions_space(self):
        """
        Constructs list with all possible actions. An action is described by a
        tuple (`elem`, `r`), indicating that the count of element `elem` will be
        set to `r`.
        """
        assert self.max_diff_elem > self.min_diff_elem
        assert self.max_atom_i > self.min_atom_i
        valid_word_len = np.arange(self.min_atom_i, self.max_atom_i + 1)
        elements = np.arange(self.periodic_table)
        actions = [(elem, r) for r in valid_word_len for elem in elements]
        return actions

    def get_max_traj_len(self):
        return self.max_atoms / self.min_atom_i

    def get_mask_invalid_actions(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if forward action is
        invalid given the current state, False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done

        if done:
            return [True for _ in range(len(self.action_space) + 1)]

        mask = [False for _ in range(len(self.action_space) + 1)]
        state_elem = [i for i, e in enumerate(state) if e > 0]
        state_atoms = sum(state)

        if state_atoms < self.min_atoms:
            mask[self.eos] = True
        if len(state_elem) < self.min_diff_elem:
            mask[self.eos] = True
        if any(r not in state_elem for r in self.required_elements):
            mask[self.eos] = True

        for idx, a in enumerate(self.action_space):
            if state_atoms + a[1] > self.max_atoms:
                mask[idx] = True
            else:
                new_elem = a[0] not in state_elem
                if not new_elem:
                    mask[idx] = True
                if new_elem and len(state_elem) >= self.max_diff_elem:
                    mask[idx] = True
        return mask

    def state2oracle(self, state: List = None) -> Tensor:
        """
        Prepares a list of states in "GFlowNet format" for the oracle

        Args
        ----
        state : list
            A state

        Returns
        ----
        oracle_state : Tensor
            Tensor containing # of Li atoms, total # of atoms, and fractions of individual elements
        """
        if state is None:
            state = self.state

        # TODO: don't assume that all consecutive elements will be present
        if len(state) < 3:
            raise ValueError(
                "state2oracle needs to return the number of Li atoms, but Li count not present in the state."
            )

        # state[2] == Li atom count, assuming consecutive elements
        return torch.Tensor([state[2], sum(state)] + [x / sum(state) for x in state])

    def state2readable(self, state=None):
        """
        Transforms the state, represented as a list of elements' counts, into a
        human-readable dict mapping elements' names to their corresponding counts.

        Example:
            state: [2, 0, 1, 0]
            self.alphabet: {0: "Li", 1: "O", 2: "C", 3: "S"}
            output: {"Li": 2, "C": 1}
        """
        if state is None:
            state = self.state
        readable = {self.alphabet[i]: s_i for i, s_i in enumerate(state) if s_i > 0}
        return readable

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.

        Example:
            readable: {"Li": 2, "C": 1} OR {"Li": 2, "C": 1, "O": 0, "S": 0}
            self.alphabet: {0: "Li", 1: "O", 2: "C", 3: "S"}
            output: [2, 0, 1, 0]
        """
        state = [0 for _ in range(self.periodic_table)]
        rev_alphabet = {v: k for k, v in self.alphabet.items()}
        for k, v in readable.items():
            state[rev_alphabet[k]] = v
        return state

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = [0 for _ in range(self.periodic_table)]
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None, actions=None):
        """
        Determines all parents and actions that lead to a state.

        Args
        ----
        state : list
            Representation of a state as a list of length self.periodic_table,
            where i-th value contains the count of atoms for i-th element, from 0 to
            self.max_atoms_i.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        actions : None
            Ignored

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
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                if state[a[0]] == a[1] > 0:
                    parent = state.copy()
                    parent[a[0]] -= a[1]
                    parents.append(parent)
                    actions.append(idx)
        return parents, actions

    def step(self, action_idx):
        """
        Executes step given an action index
        If action_idx is smaller than eos (no stop), add action to next
        position.
        See: step_daug()
        See: step_chain()
        Args
        ----
        action_idx : int
            Index of action in the action space. a == eos indicates "stop action"
        Returns
        -------
        self.state : list
            The sequence after executing the action
        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # If only possible action is eos, then force eos
        if sum(self.state) == self.max_atoms:
            self.done = True
            self.n_actions += 1
            return self.state, [self.eos], True
        # If action is not eos, then perform action
        if action_idx != self.eos:
            atomic_number, num = self.action_space[action_idx]
            state_next = self.state[:]
            state_next[atomic_number] = num
            if sum(state_next) > self.max_atoms:
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
            return self.state, action_idx, valid
        # If action is eos, then perform eos
        else:
            if sum(self.state) < self.min_atoms:
                valid = False
            else:
                nums_charges = [
                    (num, self.oxidation_states[i])
                    for i, num in enumerate(self.state)
                    if num > 0
                ]
                sum_diff_elem = []
                for n, c in nums_charges:
                    charge_sums = []
                    for c_i in itertools.product(c, repeat=n):
                        charge_sums.append(sum(c_i))
                    sum_diff_elem.append(np.unique(charge_sums))
                poss_charge_sum = [
                    sum(combo) == 0 for combo in itertools.product(*sum_diff_elem)
                ]
                if any(poss_charge_sum):
                    self.done = True
                    valid = True
                    self.action += 1
                else:
                    valid = False
            return self.state, self.eos, valid
