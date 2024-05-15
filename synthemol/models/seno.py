from typing import List
from rdkit import Chem
import torch
import numpy as np
from gneprop.rewards import load_model, predict
from wurlitzer import pipes
from pydantic import Field
from pydantic.dataclasses import dataclass
import os

@dataclass
class Parameters:
    batch_size: int = 256
    checkpoint_path: str = "/workspace/Tyers/benchmark/Mpro-GFN/src/gflownet/gneprop_weights/epoch=31-step=2272.ckpt"
    log_dir = "./logs_seno"


class SenoProxy():
    def __init__(self, params: Parameters):
        self.batch_size = params.batch_size
        self.model = load_model(params.checkpoint_path)
        self.log_dir = params.log_dir

    def __call__(self, smiles: str) -> np.array:
        with pipes():
            scores = (
                predict(
                    self.model,
                    [smiles],
                    batch_size=self.batch_size,
                    gpus=1,
                )
                * 100
            ).tolist()

            with open(os.path.join(self.log_dir, "visited.txt"), 'a') as file:
                # Write each molecule and its score to the file
                for molecule, score in zip([smiles], scores):
                    file.write(f"{molecule}, {score}\n")

        return scores[0]