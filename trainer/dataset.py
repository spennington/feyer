import pandas as pd
import torch
from torch.utils.data import Dataset

class CrosswordClueAnswersDataset(Dataset):
    """Crossword clues and answers dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with clues and answers.
        """
        self.values = pd.read_csv(csv_file, keep_default_na=False).values
        
    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.values[idx, :]
        return data
