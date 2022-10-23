from torch.utils import data
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root):
        # pred_dirs = os.listdir(pred_root)
        # label_dirs = os.listdir(label_root)

        name_list = []
        # for idir in pred_dirs:
        #     if idir in label_dirs:
        #         pred_names = os.listdir(os.path.join(pred_root, idir))
        #         label_names = os.listdir(os.path.join(label_root, idir))
        #         for iname in pred_names:
        #             if iname in label_names:
        #                 dir_name_list.append(os.path.join(idir, iname))

        pred_names = os.listdir(pred_root)
        label_names = os.listdir(label_root)
        for iname in pred_names:
            if iname in label_names:
                name_list.append(iname)

        self.image_path = list(
            map(lambda x: os.path.join(pred_root, x), name_list))
        self.label_path = list(
            map(lambda x: os.path.join(label_root, x), name_list))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
