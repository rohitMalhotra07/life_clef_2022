import torch
import pandas as pd
from torch.utils.data import Dataset
from GLC.data_loading.common import load_patch
import numpy as np


class MultiChannelImgData(Dataset):
    def __init__(self, df, env_vec_columns, landcover_mapping, data_path):
        self.df = df
        self.env_vec_columns = env_vec_columns
        self.landcover_mapping = landcover_mapping
        self.data_path = data_path

    def __getitem__(self, idx):
        x  = np.array(self.df.iloc[idx][self.env_vec_columns].values.tolist())
        # print(x)
        target = self.df.iloc[idx]['species_id']
        obs_id = self.df.iloc[idx].observation_id
        
        patch = load_patch(obs_id, self.data_path, landcover_mapping=self.landcover_mapping, data = 'rgb')
        
        
        # img_data = torch.concat([
        #                             torch.tensor(patch[0]).permute(2,0,1),
        #                             torch.tensor(patch[1]).unsqueeze(dim=0),
        #                             torch.tensor(patch[2]).unsqueeze(dim=0),
        #                             torch.tensor(patch[3]).unsqueeze(dim=0)
        #                         ],
        #                         dim = 0
        #                         )
        
        img_data = torch.tensor(patch[0]).permute(2,0,1)
        
        # print(img_data.dtype, x.dtype, target.dtype)
        
        return img_data.float(), torch.tensor(x).float(), torch.tensor(target)

    def __len__(self):
        return len(self.df)