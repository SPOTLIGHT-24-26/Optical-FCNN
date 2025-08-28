import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
from collections import defaultdict
import zipfile
import os
import numpy as np
import zipfile


class CocoPanoptic(Dataset):
    def __init__(self, root='data/coco', split='val', img_size=70, grayscale=True, download=False, mode = 'fourier', img_target=30):
        '''
        Loads the COCO panoptic dataset, that contains both stuff and things segmentations. 
        Inputs:
            img_size: the dimension of the image the model will use as input (by cropping original images from the dataset)
            mode: whether the model expects the inputs in fourier or spatial domain
            img_target: the dimension the model will give as output (model predicts categories of a subset of pixels from the input image)
            download: whether to download the dataset (data/coco should already be created empty)
        '''

        assert split in ['train', 'val'], "split must be 'train' or 'val'"
        self.root = root
        self.mode = mode
        self.split = split
        self.grayscale = grayscale
        self.img_size = img_size
        self.img_dir = os.path.join(root, f'{split}2017')
        self.ann_dir = os.path.join(root, 'annotations', f'panoptic_{split}2017')
        self.ann_json = os.path.join(root, 'annotations', f'panoptic_{split}2017.json')

        if download:
            self._download_data()

        # load annotations
        with open(self.ann_json, 'r') as f:
            self.anns = json.load(f)
        
        # get image metadata
        self.images = {img['id']: img for img in self.anns['images']} # images
        self.segments = {ann['image_id']: ann for ann in self.anns['annotations']} # annotation for each image
        self.categories = self.anns['categories'] # categories

        # map category_id to class index (we do this since category_id are not sequential numbers)
        self.cat_id_to_class_idx = {cat['id']: i+1 for i, cat in enumerate(self.categories)} # category 0 will be for unlabeled
        self.num_classes = len(self.categories) + 1
        
        self.center_crop = transforms.CenterCrop((img_size, img_size)) # crop image from trainset
        self.center_target_crop = transforms.CenterCrop((img_target, img_target)) # we predict for a smaller image (U-net)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.47038765886556555, 0.4467431241885573, 0.4073559750690107], std = [0.2788061083727918, 0.27428223938524565, 0.2886061815717444]),
            transforms.Grayscale() if grayscale else transforms.Lambda(lambda x: x),
            transforms.Normalize(mean = [0.44928963227097235], std = [0.2685102318291754]) if grayscale else transforms.Normalize((0.5,)*3, (0.5,)*3),
            transforms.Lambda(lambda x: torch.fft.fftshift(torch.fft.fft2(x, norm="forward"), dim=(-2,-1))) if mode == 'fourier' else transforms.Lambda(lambda x: x),
            self.center_crop,
            
        ])

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        ann = list(self.segments.values())[idx]
        img_info = self.images[ann['image_id']]
        #print(img_info)

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        seg_path = os.path.join(self.ann_dir, ann['file_name'])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(seg_path).convert('RGB')

        img = self.img_transform(img)
        mask = self.center_target_crop(mask)

        # convert mask to labels (convert mask to category id and then to class indexes)
        mask_tensor = torch.from_numpy(
            self._convert_mask_to_class(mask, ann['segments_info'])
        ).long()

        return img, mask_tensor

    def _convert_mask_to_class(self, mask_img, segments_info):
        '''
        For a given target mask, computes for each pixel its corresponding category_id, and from that class index that the model has to predict.
        '''
        mask_np = np.array(mask_img).astype('int32')
    
        # color represents segment id
        packed = mask_np[:, :, 0] + 256 * mask_np[:, :, 1] + 256**2 * mask_np[:, :, 2] # packed now has segment id for each pixel (to which segment it belongs)
        label_mask = torch.zeros(mask_np.shape[:2], dtype=torch.long) # unlabeled will stay 0

        # for each segment id calculate its class index
        seg_id_to_class = { 
            seg['id']: self.cat_id_to_class_idx[seg['category_id']] for seg in segments_info
        }

        # locate pixels of each segment and give them appropriate class index
        for seg_id, class_id in seg_id_to_class.items(): 
            label_mask[packed == seg_id] = class_id
    
        return label_mask.numpy()


    def _download_data(self):

        # Download images
        base_url = 'http://images.cocodataset.org/zips/'
        ann_url = 'http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip'
        img_zip = f'{self.split}2017.zip'

        if not os.path.exists(self.img_dir):
            download_and_extract_archive(base_url + img_zip, download_root=self.root)

        # Download annotations
        ann_dir = os.path.join(self.root, 'annotations')
        if not os.path.exists(ann_dir):
            download_and_extract_archive(ann_url, download_root=self.root)

        # Download panoptic masks
        pano_imgs = os.path.join(self.root, 'annotations', f'panoptic_{self.split}2017.zip')
        os.makedirs(ann_dir, exist_ok=True)
        with zipfile.ZipFile(pano_imgs, 'r') as zip_ref:
            zip_ref.extractall(ann_dir)

        print("finished dl")
       

    def get_num_classes(self):
        return self.num_classes

    
