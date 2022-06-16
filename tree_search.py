"""Contains classes and functions for performing a tree search to generate molecules combinatorially."""
import itertools
import json
import math
import pickle
from functools import cache, cached_property, partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from sklearn.ensemble import RandomForestClassifier
from tap import Tap
from tqdm import trange

from morgan_fingerprint import compute_morgan_fingerprint
from real_reactions import Reaction, REAL_REACTIONS, SYNNET_REACTIONS


class Args(Tap):
    model_path: Path  # Path to PKL file containing a RandomForestClassifier trained on Morgan fingerprints.
    fragment_path: Path  # Path to CSV file containing molecular building blocks.
    fragment_to_score_path: Path  # Path to JSON file containing a dictionary mapping fragments to prediction scores.
    reagent_to_fragments_path: Path  # Path to JSON file containing a dictionary mapping from reagents to fragments.
    save_path: Path  # Path to CSV file where generated molecules will be saved.
    search_type: Literal['random', 'greedy', 'mcts']  # Type of search to perform.
    smiles_column: str = 'smiles'  # Name of the column containing SMILES.
    max_reactions: int = 3  # Maximum number of reactions that can be performed to expand fragments into molecules.
    n_rollout: int = 100  # The number of times to run the tree search.
    c_puct: float = 10.0  # The hyperparameter that encourages exploration.
    num_expand_nodes: Optional[int] = None  # The number of tree nodes to expand when extending the child nodes in the search tree.
    synnet_rxn: bool = False  # Whether to include SynNet reactions in addition to REAL reactions.


class TreeNode:
    """A node in a tree search representing a step in the molecule construction process."""

    def __init__(self,
                 c_puct: float,
                 scoring_fn: Callable[[str], float],
                 node_id: Optional[int] = None,
                 fragments: Optional[tuple[str]] = None,
                 construction_log: Optional[tuple[dict[str, Any]]] = None) -> None:
        """Initializes the TreeNode object.

        :param c_puct: The hyperparameter that encourages exploration.
        :param scoring_fn: A function that takes as input a SMILES representing a molecule and returns a score.
        :param node_id: The ID of the node, which should correspond to the order in which the node was visited.
        :param fragments: A tuple of SMILES containing the fragments for the next reaction.
                         The first element is the currently constructed molecule while the remaining elements
                         are the fragments that are about to be added.
        :param construction_log: A tuple of dictionaries containing information about each reaction.
        """
        self.c_puct = c_puct
        self.scoring_fn = scoring_fn
        self.node_id = node_id
        self.fragments = fragments if fragments is not None else tuple()
        self.construction_log = construction_log if construction_log is not None else tuple()
        # TODO: maybe change sum (really mean) to max since we don't care about finding the best leaf node, just the best node along the way?
        self.W = 0.0  # The sum of the leaf node values for leaf nodes that descend from this node.
        self.N = 0  # The number of leaf nodes that have been visited from this node.
        self.children: list[TreeNode] = []

    @classmethod
    def compute_score(cls, fragments: tuple[str], scoring_fn: Callable[[str], float]) -> float:
        """Computes the score of the fragments.

        :param fragments: A tuple of SMILES containing the fragments for the next reaction.
                         The first element is the currently constructed molecule while the remaining elements
                         are the fragments that are about to be added.
        :param scoring_fn: A function that takes as input a SMILES representing a molecule and returns a score.
        :return: The score of the fragments.
        """
        # TODO: change this!!! to weight the fragments differently
        return sum(scoring_fn(fragment) for fragment in fragments) / len(fragments) if len(fragments) > 0 else 0.0

    @cached_property
    def P(self) -> float:
        """The property score of this node. (Note: The value is cached so it assumes the node is immutable.)"""
        return self.compute_score(fragments=self.fragments, scoring_fn=self.scoring_fn)

    def Q(self) -> float:
        """Value that encourages exploitation of nodes with high reward."""
        return self.W / self.N if self.N > 0 else 0.0

    # TODO: is it okay to use math.sqrt(1 + n) or should it go back to math.sqrt(n)?
    def U(self, n: int) -> float:
        """Value that encourages exploration of nodes with few visits."""
        return self.c_puct * self.P * math.sqrt(1 + n) / (1 + self.N)

    @property
    def num_fragments(self) -> int:
        """Returns the number of fragments for the upcoming reaction.

        :return: The number of fragments for the upcoming reaction.
        """
        return len(self.fragments)

    @property
    def num_reactions(self) -> int:
        """Returns the number of reactions used so far to generate the current molecule.

        :return: The number of reactions used so far to generate the current molecule.
        """
        return len(self.construction_log)

    def __hash__(self) -> int:
        return hash(self.fragments)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TreeNode):
            return False

        return self.fragments == other.fragments


class TreeSearcher:
    """A class that runs a tree search to generate high scoring molecules."""

    def __init__(self,
                 search_type: Literal['random', 'greedy', 'mcts'],
                 fragment_to_index: dict[str, int],
                 reagent_to_fragments: dict[str, list[str]],
                 max_reactions: int,
                 scoring_fn: Callable[[str], float],
                 n_rollout: int,
                 c_puct: float,
                 num_expand_nodes: int,
                 reactions: list[Reaction]) -> None:
        """Creates the TreeSearcher object.

        :param search_type: Type of search to perform.
        :param fragment_to_index: A dictionary mapping fragment SMILES to their index in the fragment file.
        :param reagent_to_fragments: A dictionary mapping a reagent SMARTS to all fragment SMILES that match that reagent.
        :param max_reactions: The maximum number of reactions to use to construct a molecule.
        :param scoring_fn: A function that takes as input a SMILES representing a molecule and returns a score.
        :param n_rollout: The number of times to run the tree search.
        :param c_puct: The hyperparameter that encourages exploration.
        :param num_expand_nodes: The number of tree nodes to expand when extending the child nodes in the search tree.
        :param reactions: A list of chemical reactions that can be used to combine molecular fragments.
        """
        self.search_type = search_type
        self.fragment_to_index = fragment_to_index
        self.reagent_to_fragments = reagent_to_fragments
        self.all_fragments = list(dict.fromkeys(
            fragment
            for reagent in reagent_to_fragments
            for fragment in self.reagent_to_fragments[reagent]
        ))
        self.max_reactions = max_reactions
        self.scoring_fn = scoring_fn
        self.n_rollout = n_rollout
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes
        self.reactions = reactions
        self.rng = np.random.default_rng(seed=0)

        self.TreeNodeClass = partial(TreeNode, c_puct=c_puct, scoring_fn=scoring_fn)
        self.root = self.TreeNodeClass(node_id=1)
        self.state_map: dict[TreeNode, TreeNode] = {self.root: self.root}

    def random_choice(self, array: list[Any], size: Optional[int] = None, replace: bool = True) -> Any:
        if size is None:
            return array[self.rng.integers(len(array))]

        return [
            array[i] for i in self.rng.choice(len(array), size=size, replace=replace)
        ]

    # TODO: documentation
    @classmethod
    def get_reagent_matches_per_mol(cls, reaction: Reaction, fragments: tuple[str]) -> list[list[int]]:
        return [
            [
                reagent_index
                for reagent_index, reagent in enumerate(reaction.reagents)
                if reagent.has_substruct_match(fragment)
            ]
            for fragment in fragments
        ]

    # TODO: documentation
    def get_next_fragments(self, fragments: tuple[str]) -> list[str]:
        # For each reaction these fragments can participate in, get all the unfilled reagents
        unfilled_reagents = set()
        for reaction in self.reactions:
            reagent_indices = set(range(reaction.num_reagents))

            # Skip reaction if there's no room to add more reagents
            if len(fragments) >= reaction.num_reagents:
                continue

            # For each mol, get a list of indices of reagents it matches
            reagent_matches_per_mol = self.get_reagent_matches_per_mol(reaction=reaction, fragments=fragments)

            # Loop through products of reagent indices that the mols match to
            # and for each product, if it matches to all separate reagents,
            # then include the missing reagents in the set of unfilled reagents
            for matched_reagent_indices in itertools.product(*reagent_matches_per_mol):
                matched_reagent_indices = set(matched_reagent_indices)

                if len(matched_reagent_indices) == len(fragments):
                    unfilled_reagents |= {reaction.reagents[index] for index in
                                          reagent_indices - matched_reagent_indices}

        if len(unfilled_reagents) == 0:
            return []

        # Get all the fragments that match the other reagents
        # TODO: cache this? or allow duplicates?
        available_fragments = list(dict.fromkeys(
            fragment
            for reagent in sorted(unfilled_reagents, key=lambda reagent: reagent.smarts)
            for fragment in self.reagent_to_fragments[reagent.smarts]
        ))

        return available_fragments

    def get_reactions_for_fragments(self, fragments: tuple[str]) -> list[tuple[Reaction, dict[str, int]]]:
        matching_reactions = []

        for reaction in self.reactions:
            if len(fragments) != reaction.num_reagents:
                continue

            # For each mol, get a list of indices of reagents it matches
            reagent_matches_per_mol = self.get_reagent_matches_per_mol(reaction=reaction, fragments=fragments)

            # Include every assignment of fragments to reagents that fills all the reagents
            for matched_reagent_indices in itertools.product(*reagent_matches_per_mol):
                if len(set(matched_reagent_indices)) == reaction.num_reagents:
                    fragment_to_reagent_index = dict(zip(fragments, matched_reagent_indices))
                    matching_reactions.append((reaction, fragment_to_reagent_index))

        return matching_reactions

    def run_all_reactions(self, node: TreeNode) -> list[TreeNode]:
        matching_reactions = self.get_reactions_for_fragments(fragments=node.fragments)

        product_nodes = []
        product_set = set()
        for reaction, fragment_to_reagent_index in matching_reactions:
            # Put fragments in the right order for the reaction
            fragments = sorted(node.fragments, key=lambda frag: fragment_to_reagent_index[frag])

            # Run reaction
            products = reaction.run_reactants(fragments)

            if len(products) == 0:
                raise ValueError('Reaction failed to produce products.')

            assert all(len(product) == 1 for product in products)

            # Convert product mols to SMILES (and remove Hs)
            products = [Chem.MolToSmiles(Chem.RemoveHs(product[0])) for product in products]

            # Filter out products that have already been created and deduplicate
            products = list(dict.fromkeys(product for product in products if product not in product_set))
            product_set |= set(products)

            # Create reaction log
            reaction_log = {
                'reaction_id': reaction.id,
                'reagent_ids': [self.fragment_to_index.get(fragment, -1) for fragment in fragments],
                'reagent_smiles': [fragment for fragment in fragments]
            }

            product_nodes += [
                self.TreeNodeClass(
                    fragments=(product,),
                    construction_log=node.construction_log + (reaction_log,)
                )
                for product in products
            ]

        return product_nodes

    def expand_node(self, node: TreeNode) -> list[TreeNode]:
        # Run all valid reactions on the current fragments to generate new molecules
        new_nodes = self.run_all_reactions(node=node)

        # Add all possible next fragments to the current fragment
        if node.num_fragments == 0:
            next_fragments = self.all_fragments
        else:
            next_fragments = self.get_next_fragments(fragments=node.fragments)

        # Limit the number of next fragments to expand
        if self.num_expand_nodes is not None and len(next_fragments) > self.num_expand_nodes:
            next_fragments = self.random_choice(next_fragments, size=self.num_expand_nodes, replace=False)

        # Convert next fragment tuples to TreeNodes
        new_nodes += [
            self.TreeNodeClass(
                fragments=node.fragments + (next_fragment,),
                construction_log=node.construction_log
            )
            for next_fragment in next_fragments
        ]

        return new_nodes

    def rollout(self, node: TreeNode) -> float:
        """Performs a Monte Carlo Tree Search rollout.

        :param node: An TreeNode representing the root of the MCTS search.
        :return: The value (reward) of the rollout.
        """
        # Stop the search if we've reached the maximum number of reactions
        if node.num_reactions >= self.max_reactions:
            return node.P

        # TODO: prevent exploration of a subtree that has been fully explored
        # Expand if this node has never been visited
        if len(node.children) == 0:
            # Expand the node both by running reactions with the current fragments and adding new fragments
            new_nodes = self.expand_node(node=node)

            if len(new_nodes) == 0:
                if node.num_fragments == 1:
                    return node.P
                else:
                    raise ValueError('Failed to expand a partially expanded node.')

            # Add the expanded nodes as children
            child_set = set()
            for new_node in new_nodes:

                # Check whether this node was already added as a child
                if new_node not in child_set:

                    # Check the state map and merge with an existing node if available
                    new_node = self.state_map.setdefault(new_node, new_node)

                    # Assign node ID as order in which the node was added
                    if new_node.node_id is None:
                        new_node.node_id = len(self.state_map)

                    # Add the new node as a child of the tree node
                    node.children.append(new_node)
                    child_set.add(new_node)

        # TODO: think more about balancing reaction nodes vs additional fragment nodes

        # Select a child node based on the search type
        if self.search_type == 'random':
            selected_node = self.random_choice(node.children)

        elif self.search_type == 'greedy':
            # Randomly sample from top k
            # TODO: better method?
            top_k = 100
            sorted_children = sorted(node.children, key=lambda child: child.P, reverse=True)
            selected_node = self.random_choice(sorted_children[:top_k])

        elif self.search_type == 'mcts':
            sum_count = sum(child.N for child in node.children)
            selected_node = max(node.children, key=lambda child: child.Q() + child.U(n=sum_count))

        else:
            raise ValueError(f'Search type "{self.search_type}" is not supported.')

        # Unroll the child node
        v = self.rollout(node=selected_node)
        selected_node.W += v
        selected_node.N += 1

        return v

    def search(self) -> list[TreeNode]:
        """Runs the tree search.

        :return: A list of TreeNode objects sorted from highest to lowest reward.
        """
        for _ in trange(self.n_rollout):
            self.rollout(node=self.root)

        # Get all the nodes representing fully constructed molecules
        nodes = [node for _, node in self.state_map.items() if node.num_fragments == 1]

        # Sort by highest reward and break ties by preferring less unmasked results
        nodes = sorted(nodes, key=lambda node: node.P, reverse=True)

        return nodes


def save_molecules(nodes: list[TreeNode], save_path: Path) -> None:
    # Create save directory
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert construction logs from lists to dictionaries
    construction_dicts = []
    max_reaction_num = 0
    reaction_num_to_max_reagent_num = {}

    for node in nodes:
        construction_dict = {'num_reactions': len(node.construction_log)}
        max_reaction_num = max(max_reaction_num, len(node.construction_log))

        for reaction_index, reaction_log in enumerate(node.construction_log):
            reaction_num = reaction_index + 1
            construction_dict[f'reaction_{reaction_num}_id'] = reaction_log['reaction_id']

            reaction_num_to_max_reagent_num[reaction_num] = max(
                reaction_num_to_max_reagent_num.get(reaction_num, 0),
                len(reaction_log['reagent_ids'])
            )

            for reagent_index, (reagent_id, reagent_smiles) in enumerate(zip(reaction_log['reagent_ids'],
                                                                             reaction_log['reagent_smiles'])):
                reagent_num = reagent_index + 1
                construction_dict[f'reagent_{reaction_num}_{reagent_num}_id'] = reagent_id
                construction_dict[f'reagent_{reaction_num}_{reagent_num}_smiles'] = reagent_smiles

        construction_dicts.append(construction_dict)

    # Specify column order for CSV file
    columns = ['smiles', 'node_id', 'num_visits', 'score', 'Q_value', 'num_reactions']

    for reaction_num in range(1, max_reaction_num + 1):
        columns.append(f'reaction_{reaction_num}_id')

        for reagent_num in range(1, reaction_num_to_max_reagent_num[reaction_num] + 1):
            columns.append(f'reagent_{reaction_num}_{reagent_num}_id')
            columns.append(f'reagent_{reaction_num}_{reagent_num}_smiles')

    # Save data
    data = pd.DataFrame(
        data=[
            {
                'smiles': node.fragments[0],
                'node_id': node.node_id,
                'num_visits': node.N,
                'score': node.P,
                'Q_value': node.Q(),
                **construction_dict
            }
            for node, construction_dict in zip(nodes, construction_dicts)
        ],
        columns=columns
    )
    data.to_csv(save_path, index=False)


# TODO: change this scoring function!!!
@cache
def log_p_score(smiles: str) -> float:
    return MolLogP(Chem.MolFromSmiles(smiles))


def run_tree_search(args: Args) -> None:
    """Generate molecules combinatorially by performing a tree search."""
    if args.synnet_rxn:
        reactions = REAL_REACTIONS + SYNNET_REACTIONS
    else:
        reactions = REAL_REACTIONS

    # Load model
    with open(args.model_path, 'rb') as f:
        model: RandomForestClassifier = pickle.load(f)

    # Load fragments
    fragments = pd.read_csv(args.fragment_path)[args.smiles_column]

    # Map fragments to indices
    fragment_to_index = {}

    for index, fragment in enumerate(fragments):
        if fragment not in fragment_to_index:
            fragment_to_index[fragment] = index

    # Load mapping from SMILES to scores
    with open(args.fragment_to_score_path) as f:
        fragment_to_score: dict[str, float] = json.load(f)

    # Load mapping from reagents to fragments
    with open(args.reagent_to_fragments_path) as f:
        reagent_to_fragments: dict[str, list[str]] = json.load(f)

    # Define scoring function
    @cache
    def model_scoring_fn(smiles: str) -> float:
        if smiles not in fragment_to_score:
            fingerprint = compute_morgan_fingerprint(smiles)
            prob = model.predict_proba([fingerprint])[0, 1]
        else:
            prob = fragment_to_score[smiles]

        return prob

    # Set up TreeSearchRunner
    tree_searcher = TreeSearcher(
        search_type=args.search_type,
        fragment_to_index=fragment_to_index,
        reagent_to_fragments=reagent_to_fragments,
        max_reactions=args.max_reactions,
        scoring_fn=model_scoring_fn,
        n_rollout=args.n_rollout,
        c_puct=args.c_puct,
        num_expand_nodes=args.num_expand_nodes,
        reactions=reactions
    )

    # Search for molecules
    nodes = tree_searcher.search()

    # Save generated molecules
    save_molecules(nodes=nodes, save_path=args.save_path)


if __name__ == '__main__':
    run_tree_search(Args().parse_args())
