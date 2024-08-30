import numpy as np
import os
import io
import cv2
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)

import torch

import sys
sys.path.append('/fs/cfar-projects/actionloc/bounce_back/InternVideo/InternVideo2/multi_modality')
sys.path.append('/fs/cfar-projects/actionloc/bounce_back/InternVideo/InternVideo2/multi_modality/demo')

from config import (Config,
                    eval_dict_leaf)
import importlib
import utils
importlib.reload(utils)
import pandas as pd
import hickle
import argparse

from utils import (retrieve_text,
                  _frame_from_video,
                  setup_internvideo2,
                  get_text_features, get_video_features)


if __name__ == '__main__':
    #init argparse
    parser = argparse.ArgumentParser(description='Extract features for gym99 dataset')
    parser.add_argument('--num_splits', type=int, help='Number of splits to extract features for', default=5)
    parser.add_argument('--split_id', type=int, help='Split id to extract features for', default=0)
    args = parser.parse_args()



    video = cv2.VideoCapture('/fs/cfar-projects/actionloc/bounce_back/InternVideo/InternVideo2/multi_modality/demo/example1.mp4')
    frames = [x for x in _frame_from_video(video)]
    text_candidates = ["A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
                   "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
                   "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
                   "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
                   "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
                   "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
                   "A playful dog slides down a snowy hill, wagging its tail with delight.",
                   "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
                   "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
                   "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."]
    config = Config.from_file('/fs/cfar-projects/actionloc/bounce_back/InternVideo/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py')
    config = eval_dict_leaf(config)
    model_pth = '/fs/cfar-projects/inr_analysis/semantic_inr/InternVideo2-stage2_1b-224p-f4.pt'
    config['pretrained_path'] = model_pth
    intern_model, tokenizer = setup_internvideo2(config)


    df = pd.read_csv('/fs/cfar-projects/actionloc/bounce_back/dataset_utils/gym99_with_template.csv')
    ds_df = pd.read_csv('/fs/cfar-projects/actionloc/bounce_back/point_tracking/dataset_csvs/data_to_transfer/gym99.csv')
    ds_df = ds_df[ds_df.split=='val'].reset_index(drop=True)
    feat_dump_path = '/fs/cfar-projects/video_gen_llm/pulkit/internvid_feats/finegym'
    num_in_each_split = len(ds_df)//args.num_splits
    start_idx = args.split_id*num_in_each_split
    end_idx = start_idx + num_in_each_split

    ds_df = ds_df.iloc[start_idx:end_idx]
    for idx, row in tqdm(ds_df.iterrows()):
        print(f"Processing video {idx}")
        video_path = row['video_path']
        vid_name = os.path.basename(video_path)
        vid_name = vid_name.split('.')[0]
        feats_path = os.path.join(feat_dump_path, f'{vid_name}.hkl')
        if os.path.exists(feats_path):
            continue
        video = cv2.VideoCapture(video_path)
        frames = [x for x in _frame_from_video(video)]
        feats = get_video_features(frames, model=intern_model, config=config)
        feats = feats.cpu().numpy()
        hickle.dump(feats, feats_path)


        
    



