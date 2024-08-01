import os
import json
from pathlib import Path

import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset

class CIRR(Dataset):
  

    def __init__(self, data_path, split, mode, transform, blip_transform):

        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        self._data_path, self._split, self._mode, self._transform, self._blip_transform = data_path, split, mode, transform, blip_transform

        if split == "test":
            split = "test1"
            self._split = "test1"

        self._construct_db()

    def _construct_db(self):

        self._db = []
        with open(os.path.join(self._data_path, 'cirr', 'captions', f'cap.rc2.{self._split}.json')) as f:
            self.triplets = json.load(f)
        with open(os.path.join(self._data_path, 'cirr', 'image_splits', f'split.rc2.{self._split}.json')) as f:
            self.name_to_relpath = json.load(f)
        
        if self._mode == 'relative':
            for i in range(len(self.triplets)):
                members = self.triplets[i]['img_set']['members']
                reference = self.triplets[i]['reference']
                caption = self.triplets[i]['caption']
                if self._split == 'test1':
                    pair_id = self.triplets[i]['pairid']
                    reference_image_path = os.path.join(self._data_path, self.name_to_relpath[reference])
                    self._db.append(
                          {
                            'reference_image_path': reference_image_path,
                            'reference': reference,
                            'caption': caption,
                            'members': members,
                            'pair_id': pair_id
                          }
                        )
        
        elif self._mode == 'classic':
            for i in range(len(list(self.name_to_relpath.keys()))):
                image_name = list(self.name_to_relpath.keys())[i]
                image_path = os.path.join(self._data_path, self.name_to_relpath[image_name])
                self._db.append(
                  {
                    'image_name': image_name,
                    'image_path': image_path
                  }
                )

    def __getitem__(self, index) -> dict:
        if self._mode == 'relative':
            if self._split == 'test1':
                return {
                    'reference_image': self._transform(Image.open(self._db[index]['reference_image_path'])),
                    'blip_ref_img': self._blip_transform(Image.open(self._db[index]['reference_image_path']).convert('RGB')),
                    'reference_name': self._db[index]['reference'],
                    'relative_caption': self._db[index]['caption'],
                    'group_members': self._db[index]['members'],
                    'pair_id': self._db[index]['pair_id']
                }
        
        elif self._mode == 'classic':
            return{
                'image': self._transform(Image.open(self._db[index]['image_path'])),
                'image_name': self._db[index]['image_name']
            }

    def __len__(self):
        return len(self._db)