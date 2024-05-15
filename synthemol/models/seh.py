from synthemol.models.bengio2021flow import load_original_model, mol2graph
from typing import List, Optional
import torch
import torch_geometric.data as gd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from pydantic import Field
from pydantic.dataclasses import dataclass
import os


@dataclass
class Parameters:
    model = load_original_model()
    device: str = "cuda"
    log_dir: str = "./logs_seh"
    beta: int = 8


class SEHProxy():
    def __init__(self, params: Parameters):
        self.device = params.device
        self.model = params.model
        self.log_dir = params.log_dir
        self.beta = params.beta
        self.model.to(self.device)

    def __call__(self, smiles: List[str]) -> np.array:
        mol = Chem.MolFromSmiles(smiles)
        graph = mol2graph(mol)
        smiles_to_save = [smiles]
        batch = gd.Batch.from_data_list([graph])
        batch.to(self.device)
        self.model.to(self.device)
        raw_score = self.model(batch).reshape((-1,)).data.cpu()
        raw_score[raw_score.isnan()] = 0
        score_to_save = list(raw_score)
        with open(os.path.join(self.log_dir, "visited.txt"), 'a') as file:
            # Write each molecule and its score to the file
            for molecule, score in zip(smiles_to_save, score_to_save):
                file.write(f"{molecule}, {score}\n")

        transformed_score = raw_score.clip(1e-4, 100).reshape((-1,))
        #print(f"Proxy Mean: {raw_scores.mean()}, Proxy Max: {raw_scores.max()}, Mean Reward: {transformed_scores.mean()}, Max Reward: {transformed_scores.max()}")
        return transformed_score[0]