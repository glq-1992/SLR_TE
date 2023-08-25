import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation_insert as video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, gloss_dict_T,gloss_dict_E,drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb",frame_interval=2,meaningless_frame_begin=10,meaningless_frame_end=5,meaningless_frame_begin_E=10):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.dict_T=gloss_dict_T
        self.dict_E=gloss_dict_E
        self.data_type = datatype
        self.dataset_root = '/disk2/dataset_glq/CSL_image_112'


        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        # data_txt = f"DatasetFile/chinese_combination_direct/{mode}_chinese_combination.txt"
        data_txt = f"DatasetFile/3004_new/{mode}_te_different_third.txt"  # {mode}_3004_e.txt; {mode}_signer_continuous.txt; train_3004_e_third.txt;train_signer_continuous_third.txt; test_te_continuous_third.txt
        data_txt_c = f"DatasetFile/3004_new/train_signer_third.txt"
        self.inputs_list = {'prefix':self.dataset_root}
        self.inputs_list_c = {'prefix': self.dataset_root}
        self.frame_interval=frame_interval
        self.meaningless_frame_begin=meaningless_frame_begin
        self.meaningless_frame_end=meaningless_frame_end
        self.meaningless_frame_begin_E=meaningless_frame_begin_E

        with open(data_txt) as f:
            if mode == 'train':
                for i, line in enumerate(f.readlines()):
                    line = line.replace('\n', '')
                    Tfolder, Efolder, label,Tlabel,Elabel,insertWordBefore,insertWordAfter,insertFrameBefore,insertFrameAfter = line.split('Q')
                    Tfileid = Tfolder[-13:]
                    Efileid = Efolder[-13:]
                    signer = Tfolder[-7:-3]
                    Tfolder = Tfolder + '/*.jpg'
                    Efolder = Efolder + '/*.jpg'

                    self.inputs_list[i] = dict(Tfileid=Tfileid, Efileid=Efileid, signer=signer, Tfolder=Tfolder, Efolder=Efolder, label=label, Tlabel=Tlabel,Elabel=Elabel,insertWordBefore=insertWordBefore,insertWordAfter=insertWordAfter,insertFrameBefore=insertFrameBefore,insertFrameAfter=insertFrameAfter,original_info='{}_{}'.format(Tfileid, Efileid))
                with open(data_txt_c) as f_c:
                    for j, line_c in enumerate(f_c.readlines()):
                        line_c = line_c.replace('\n', '')
                        Tfolder, Efolder, label, Tlabel, Elabel, insertWordBefore, insertWordAfter, insertFrameBefore, insertFrameAfter = line_c.split(
                            'Q')
                        Tfileid = Tfolder[-13:]
                        Efileid = Efolder[-13:]
                        signer = Tfolder[-7:-3]
                        Tfolder = Tfolder + '/*.jpg'
                        Efolder = Efolder + '/*.jpg'

                        self.inputs_list_c[j] = dict(Tfileid=Tfileid, Efileid=Efileid, signer=signer, Tfolder=Tfolder,
                                                   Efolder=Efolder, label=label, Tlabel=Tlabel, Elabel=Elabel,
                                                   insertWordBefore=insertWordBefore, insertWordAfter=insertWordAfter,
                                                   insertFrameBefore=insertFrameBefore,
                                                   insertFrameAfter=insertFrameAfter,
                                                   original_info='{}_{}'.format(Tfileid, Efileid))
            else:
                for i, line in enumerate(f.readlines()):
                    line = line.replace('\n', '')
                    folder, label = line.split('Q')
                    fileid = folder[-13:]
                    signer = folder[-7:-3]
                    folder = folder + '/*.jpg'
                    self.inputs_list[i] = dict(fileid=fileid, signer=signer, folder=folder, label=label,
                                               original_info=fileid)

        # self.inputs_list = np.load(f"./preprocess/phoenix2014/{mode}_info.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = dict([*filter(lambda x: isinstance(x[0], str) or x[0] < 10, self.inputs_list.items())])
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        if self.data_type == "video":
            if self.mode == 'train':
                Tinput_data, Einput_data, Cinput_data,label, Tlabel,Elabel,insert_word_before,insert_word_after,fi = self.read_video(idx)
                try:
                    Tinput_data, Einput_data, Cinput_data,label = self.normalize(Tvideo=Tinput_data, Evideo=Einput_data, Cvideo=Cinput_data, label=label)
                except:
                    print(fi)
                    exit()
                # Elabel=list(set(label) - set(Tlabel))
                
                # input_data, label = self.normalize(input_data, label, fi['fileid'])
                return Tinput_data, Einput_data, Cinput_data,torch.LongTensor(label),torch.LongTensor(Tlabel), torch.LongTensor(Elabel),torch.LongTensor(insert_word_before),torch.LongTensor(insert_word_after),fi['insertFrameBefore'],fi['insertFrameAfter'],self.inputs_list[idx],'train'
            else:
                input_data, label, fi = self.read_video(idx)
                input_data, label = self.normalize(Rvideo=input_data, label=label)
                # input_data, label = self.normalize(input_data, label, fi['fileid'])
                return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info'],'test'

        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info']

    def read_video(self, index, num_glosses=-1):
        # load file info
        if self.mode == 'train':
            fi = self.inputs_list[index]
            Timg_folder = fi['Tfolder']
            Eimg_folder = fi['Efolder']

            Timg_list_all = sorted(glob.glob(Timg_folder))
            Eimg_list_all = sorted(glob.glob(Eimg_folder))
            Timg_list = Timg_list_all[::self.frame_interval]
            Eimg_list = Eimg_list_all[::self.frame_interval]

            label_list = []
            T_label_list=[]
            E_label_list=[]
            insert_word_before_list=[]
            insert_word_after_list=[]

            label_word=fi['label']
            c_number=len(self.inputs_list_c)-1
            path_c=[]
            for i in range(c_number):
                if self.inputs_list_c[i]['label']==label_word:
                    path_c.append(self.inputs_list_c[i])
            fi_c=random.choice(path_c)
            cimg_folder = fi_c['Tfolder']
            cimg_list_all = sorted(glob.glob(cimg_folder))
            cimg_list = cimg_list_all[::self.frame_interval]

            for phase in fi['label'].split(" "):
                if phase == '':
                    continue
                if phase in self.dict.keys():
                    label_list.append(self.dict[phase])

            for phase in fi['Tlabel'].split(" "):
                if phase == '':
                    continue
                if phase in self.dict_T.keys():
                    T_label_list.append(self.dict_T[phase])

            for phase in fi['Elabel'].split(" "):
                if phase == '':
                    continue
                if phase in self.dict_E.keys():
                    E_label_list.append(self.dict_E[phase])

            if fi['insertWordBefore']=='0':
                insert_word_before_list.append(int(fi['insertWordBefore']))
            else:
                insert_word_before_list.append(self.dict_T[fi['insertWordBefore']])

            if fi['insertWordAfter']=='0':
                insert_word_after_list.append(int(fi['insertWordAfter']))
            else:
                insert_word_after_list.append(self.dict_T[fi['insertWordAfter']])
                    

            return [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in Timg_list], \
                   [cv2.cvtColor(cv2.imread(j), cv2.COLOR_BGR2RGB) for j in Eimg_list],\
                   [cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2RGB) for k in cimg_list], label_list, T_label_list,E_label_list,insert_word_before_list,insert_word_after_list,fi
        else:
            fi = self.inputs_list[index]
            img_folder = fi['folder']
            img_list_all = sorted(glob.glob(img_folder))
            img_list = img_list_all[::self.frame_interval]

            label_list = []
            for phase in fi['label'].split(" "):
                if phase == '':
                    continue
                if phase in self.dict.keys():
                    label_list.append(self.dict[phase])
            return [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in img_list], label_list, fi


    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, Tvideo=None, Evideo=None, Cvideo=None,Rvideo=None, label=None, file_id=None):
    
        if Tvideo:
            Tvideo, label = self.data_aug(Tvideo, label, file_id)
            Evideo, label = self.data_aug(Evideo, label, file_id)
            Cvideo, label = self.data_aug(Cvideo, label, file_id)

            # Tvideo = Tvideo.float() / 127.5 - 1
            # Evideo = Evideo.float() / 127.5 - 1
            Tvideo = Tvideo.float()/255
            Evideo = Evideo.float() / 255
            Cvideo = Cvideo.float() / 255
            return Tvideo, Evideo, Cvideo,label
        else:
            Rvideo, label = self.data_aug(Rvideo, label, file_id)
            # Rvideo = Rvideo.float() / 127.5 - 1
            Rvideo = Rvideo.float() / 255

            return Rvideo, label



    # def transform(self):
    #     if self.transform_mode == "train":
    #         print("Apply training transform.")
    #         return video_augmentation.Compose([
    #             # video_augmentation.CenterCrop(224),
    #             # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
    #             # video_augmentation.RandomCrop(224),
    #             video_augmentation.RandomHorizontalFlip(0.5),
    #             video_augmentation.ToTensor(),
    #             # video_augmentation.TemporalRescale(0.2),
    #             # video_augmentation.Resize(0.5),
    #         ])
    #     else:
    #         print("Apply testing transform.")
    #         return video_augmentation.Compose([
    #             # video_augmentation.CenterCrop(224),
    #             # video_augmentation.Resize(0.5),
    #             video_augmentation.ToTensor(),
    #         ])
    # 新版
    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                # video_augmentation.ResizeShape((224, 224)),
                video_augmentation.RandomCrop(112), ### 224 , 110, 180
                video_augmentation.RandomHorizontalFlip(0.5),
                # video_augmentation.Resize(1),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                # video_augmentation.ResizeShape((224, 224)),
                video_augmentation.CenterCrop(112), ###224 , 110, 180
                # video_augmentation.Resize(0.5),
                video_augmentation.ToTensor(),
            ])
    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        # batch = [item for item in sorted(batch, key=lambda x: len(x[0])+len(x[1]), reverse=True)]
        if batch[0][-1] == 'train':
            Tvideo, Evideo, Cvideo,label, Tlabel,Elabel,insertWordBefore,insertWordAfter,insertFrameBefore,insertFrameAfter,info,_ = list(zip(*batch))
            Tmax_len = max([len(video) for video in Tvideo])
            Emax_len = max([len(video) for video in Evideo])
            Cmax_len = max([len(video) for video in Cvideo])

            Tvideo_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in Tvideo])
            Evideo_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in Evideo])
            Cvideo_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in Cvideo])
            left_pad = 6
            Tright_pad = int(np.ceil(Tmax_len / 4.0)) * 4 - Tmax_len + 6
            Tmax_len = Tmax_len + left_pad + Tright_pad
            Tpadded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(Tmax_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in Tvideo]
            Tpadded_video = torch.stack(Tpadded_video)
            right_pad_list_T=[]
            for vid in Tvideo:
                right_pad_list_T.append(Tmax_len - len(vid) - left_pad)

            Eright_pad = int(np.ceil(Emax_len / 4.0)) * 4 - Emax_len + 6
            Emax_len = Emax_len + left_pad + Eright_pad
            Epadded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(Emax_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in Evideo]
            Epadded_video = torch.stack(Epadded_video)
            Cright_pad = int(np.ceil(Cmax_len / 4.0)) * 4 - Cmax_len + 6
            Cmax_len = Cmax_len + left_pad + Cright_pad
            Cpadded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(Cmax_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in Cvideo]
            Cpadded_video = torch.stack(Cpadded_video)


            right_pad_list_E=[]
            for vid in Evideo:
                right_pad_list_E.append(Emax_len - len(vid) - left_pad)


            label_length = torch.LongTensor([len(lab) for lab in label])
            T_label_length = torch.LongTensor([len(lab) for lab in Tlabel])
            E_label_length = torch.LongTensor([len(lab) for lab in Elabel])

            if max(label_length) == 0:
                return Tpadded_video, Epadded_video, Tvideo_length, Evideo_length, [], [], info
            else:
                padded_label = []
                padded_T_label = []
                padded_E_label = []
                padded_insertWordBefore=[]
                padded_insertWordAfter=[]
                insertFrameBeforeList=[]
                insertFrameAfterList=[]
                for lab,Tlab,Elab,insertBefore,insertAfter,insertFBefore,insertFAfter in zip(label,Tlabel,Elabel,insertWordBefore,insertWordAfter,insertFrameBefore,insertFrameAfter):
                    padded_label.extend(lab)
                    padded_T_label.extend(Tlab)
                    padded_E_label.extend(Elab)
                    padded_insertWordBefore.extend(insertBefore)
                    padded_insertWordAfter.extend(insertAfter)
                    insertFrameBeforeList.append(insertFBefore)
                    insertFrameAfterList.append(insertFAfter)
                padded_label = torch.LongTensor(padded_label)
                padded_T_label = torch.LongTensor(padded_T_label)
                padded_E_label = torch.LongTensor(padded_E_label)
                padded_insertWordBefore = torch.LongTensor(padded_insertWordBefore)
                padded_insertWordAfter = torch.LongTensor(padded_insertWordAfter)

                return Tpadded_video, Epadded_video, Cpadded_video,Tvideo_length, Evideo_length, Cvideo_length,padded_label, label_length, padded_T_label,T_label_length,padded_E_label,E_label_length,padded_insertWordBefore,padded_insertWordAfter,insertFrameBeforeList,insertFrameAfterList,right_pad_list_T,right_pad_list_E,info
        else:
            batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
            video, label, info,_ = list(zip(*batch))
            if len(video[0].shape) > 3:
                max_len = len(video[0])
                video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
                left_pad = 6
                right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
                max_len = max_len + left_pad + right_pad
                padded_video = [torch.cat(
                    (
                        vid[0][None].expand(left_pad, -1, -1, -1),
                        vid,
                        vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                    )
                    , dim=0)
                    for vid in video]
                padded_video = torch.stack(padded_video)
            else:
                max_len = len(video[0])
                video_length = torch.LongTensor([len(vid) for vid in video])
                padded_video = [torch.cat(
                    (
                        vid,
                        vid[-1][None].expand(max_len - len(vid), -1),
                    )
                    , dim=0)
                    for vid in video]
                padded_video = torch.stack(padded_video).permute(0, 2, 1)
            label_length = torch.LongTensor([len(lab) for lab in label])
            if max(label_length) == 0:
                return padded_video, video_length, [], [], info
            else:
                padded_label = []
                for lab in label:
                    padded_label.extend(lab)
                padded_label = torch.LongTensor(padded_label)
                return padded_video, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
