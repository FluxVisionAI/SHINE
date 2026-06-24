# written by Dexiang Zong
# CardioVascular Systems Imaging and Artificial Intelligence (CVS.AI)
# National Heart Research Institute Singapore (NHRIS)
# national heart center singarpore

import torch
from torch.utils.data import Dataset


class ManifoldDataset(Dataset):
    """Dataset for training SHINE neural fields.
    
    Yields 3D coordinates on the myocardial manifold and their associated
    LGE scar intensity probabilities.
    """
    def __init__(self, coordinates: torch.Tensor, intensities: torch.Tensor):
        self.coordinates = coordinates
        self.intensities = intensities

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.coordinates[idx], self.intensities[idx]
