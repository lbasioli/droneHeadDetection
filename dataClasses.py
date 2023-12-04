import os
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


GT_COLUMNS = {0:'frameID', 1:'headID', 2:'box left', 3:'box top', 4:'box width', 5:'box height', 6:'confidence_flag', 7:'class', 8:'visibility'}

#### NOT USED ####
DET_COLUMNS = {0:'frameID', 1:'headID', 2:'box left', 3:'box top', 4:'box width', 5:'box height', 6:'confidence', 7:'confidence_flag', 8:'class', 9:'visibility'}
CLASSES = {0:'Pedestrian', 1:'Person on Vehicle', 2:'Static', 3:'Ignore'}
#################

# A comment on datasets: My initial idea was to use the {challange test dataset} as {my task validation dataset} but only later I concluded that det files should not be
# used. Although the original authors' results in det file seem to be good enough in the practical sense, theroetically they could introduce bias coming from a flaw in
# Headhunter, Headhunter-T or the training proccess written by the original authors. Therefore only the data that comes coupled with ground truth (labeled data) is used

# This class is used to save all paths to data and annotations
class Data:
    # Everything is done in __init__ with the help of some getters
    def __init__(self, data_dirs, det_files = None, gt_files = None, augmentation=False):
        self.data_dirs = data_dirs
        if det_files:
            self.det_files = det_files
            self.detection = self.get_detection()
        if gt_files:
            self.gt_files = gt_files
            self.ground_truth = self.get_ground_truth()
        self.frame_list = self.get_frame_list()
    
    # Returns a list of lists - e.g. 2 scenes, one having 100 frames and another 200 will return [[100 strings] [200 strings]] where each string is a path to a
    # frame found in each data_dir given to the Data constructor
    def get_frame_list(self):
        frames = []
        for directory in self.data_dirs:
            frames += sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')])
        return frames

    ####### NOT USED ########
    def get_detection(self):
        dets = pd.DataFrame(columns = range(10))
        num_of_frames = 0

        for det in self.det_files:
            temp = pd.read_csv(det, header = None)
            temp[0] += num_of_frames
            num_of_frames = temp[0].max()
            dets = pd.concat([dets, temp])

        dets.reset_index(drop = True, inplace = True)
        dets.rename(columns = DET_COLUMNS, inplace = True)
        dets['box right'] = dets['box left'] + dets['box width']
        dets['box bot'] = dets['box top'] + dets['box height']
        return dets[['frameID', 'box left', 'box top', 'box right', 'box bot']].copy()
    ##########################

    # annotations getter. also transforms data format from CroHD standard to ResNet-50 standard
    def get_ground_truth(self):
        gts = pd.DataFrame(columns = range(9))
        num_of_frames = 0

        for gt in self.gt_files:
            temp = pd.read_csv(gt, header = None)
            temp[0] += num_of_frames
            num_of_frames = temp[0].max()
            gts = pd.concat([gts, temp])

        gts.reset_index(drop = True, inplace = True)
        gts.rename(columns = GT_COLUMNS, inplace = True)
        gts['box right'] = gts['box left'] + gts['box width']
        gts['box bot'] = gts['box top'] + gts['box height']
        return gts[['frameID', 'box left', 'box top', 'box right', 'box bot']].copy()


# This class inherits from torch.utils.data.Dataset so that we can use it to create torch.utils.data.DataLoader that will be used to efficiently train, evaluate
# and visualise the model. We must define __init__, __len__ and __getitem__ for it to be able to communicate with the torchvision fasterrcnn_resnet50_fpn class.
class headDetectionDataset(Dataset):
    # constructor inputs - data: Data class defined above
    #                    - transform: a function defined outside, recommended is torchvision.transforms.Compose([custom pipeline])
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data.frame_list)

    # This class does not save all data as a member variable, that would be insane. Instead, it only stores all paths in the Data object to save RAM. That is why we need
    # this function to translate paths related to a single batch to data and annotations that now have a size we can handle.
    # Input - frame index (0-based), cumulative across scenes (This is handled in Data class getters by += operator in get_frame_list and num_of_frames in get_ground_truth)
    # Return - tuple of data and target formatted for forward propagation of fasterrcnn_resnet50_fpn
    def __getitem__(self, idx):
        # get frame:
        image = cv2.imread(self.data.frame_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads images in BGR, convert to RGB        
        image = image.astype(np.float32) / 255.0        # Normalize image to the range [0, 1]
        image = np.transpose(image, (2, 0, 1))          # Transpose image dimensions from [H, W, C] to [C, H, W]
        
        # get target:
        boxes, labels = self.get_annotations(idx + 1)   # Index in the dataset is 1-based
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
        }

        return torch.tensor(image), target

    # Function used by __getitem__ to transform annotations to required format
    def get_annotations(self, frame_no):
        detections_frame = self.data.ground_truth[self.data.ground_truth['frameID'] == frame_no]   # Filter detections for the specific frame
        boxes = detections_frame[['box left', 'box top', 'box right', 'box bot']].values           # Format used by fasterrcnn_resnet50_fpn
        labels = [0] * len(boxes)   # we disregard classification to pedestrians, people in vehicles etc. and put every label to the same class
        return boxes, labels

    # Function used by DataLoader to prepare single batch to a nice format used for looping
    def collate_fn(self, batch):
        images, targets = zip(*batch)

        # Filter out empty targets (frames without detections) if any
        targets = [target for target in targets if target['boxes'].size(0) > 0]
        
        targets_padded = [     # Format used by fasterrcnn_resnet50_fpn
            {
                'boxes': target['boxes'],
                'labels': target['labels'],
            }
            for target in targets
        ]
        return torch.stack(images), targets_padded