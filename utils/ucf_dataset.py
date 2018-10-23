import os
import random
import numpy as np
import PIL.Image as Image
import torch
import torch.utils.data as data

class UCF101(data.Dataset):
    base_folder = os.path.join(os.path.dirname(__file__), 'lists')
    train_list_file = 'train_ucf.list'
    test_list_file = 'test_ucf.list'

    def __init__(self, train=True, transform=None, target_transform=None, num_frames_per_clip=16):
        self.train = train # train set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.num_frames_per_clip = num_frames_per_clip

        if self.train:
            list_file = os.path.join(self.base_folder, self.train_list_file)
        else:
            list_file = os.path.join(self.base_folder, self.test_list_file)

        self.lists = open(list_file, 'r').read().splitlines()
    
    def __getitem__(self, index):
        line = self.lists[index].split()
        clip_path = line[0]
        target = int(line[1])

        clips = self._load_clip_data(clip_path)
        clips = torch.stack(clips, 1)

        return clips, target

    def __len__(self):
        return len(self.lists)

    def _load_clip_data(self, clip_path):
        return_array = []
        for root, dirs, files in os.walk(clip_path):
            files = sorted(files)
            num_files = len(files)
            if num_files < self.num_frames_per_clip:
                start_id = 0
                ids = np.arange(0, self.num_frames_per_clip)
                ids[num_files:] = num_files - 1
            elif num_files == self.num_frames_per_clip:
                start_id = 0
                ids = np.arange(0, self.num_frames_per_clip)
            else:
                start_id = random.randint(0, num_files - self.num_frames_per_clip - 1)
                ids = np.arange(start_id, start_id + self.num_frames_per_clip)
            
            # step = num_files / self.num_frames_per_clip
            # ids = np.arange(0, num_files, step, dtype=np.int32)
            # last_id = ids[-1]
            # offset = random.randint(0, num_files - last_id - 1)
            # ids += offset

            for i in ids:
                image_path = os.path.join(clip_path, files[i])
                img = Image.open(image_path)
                if self.transform is not None:
                    img = self.transform(img)
                return_array.append(img)
            
        return return_array
