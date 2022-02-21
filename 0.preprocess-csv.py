# %%
import pandas as pd
import os
import sys
import numpy as np
from glob import glob

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

challenge_names = [
    "AU_Detection_Challenge",
    "EXPR_Classification_Challenge",
    "MTL_Challenge",
    "VA_Estimation_Challenge",
]

# challenge_cols = ['AU_Detection_Cha']
splits = ["Train_Set", "Validation_Set"]

PATH = "/mnt/DATA2/congvm/Affwild2/Annotations/"
CROP_PATHS = "/mnt/DATA2/congvm/Affwild2/cropped_aligned/"
EMOTION_MAP = {
    -1: "Ignored",
    0: "Neutral",
    1: "Anger",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Sadness",
    6: "Surprise",
    7: "Other",
}


# %%
def read_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split(",") for line in lines]
        return lines


def load_image(img_path):
    img_arr = cv2.imread(img_path)
    return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)


# %%
# Load Training
fails = {}
success = {}

challenge_names = ["MTL_Challenge"]
for chname in challenge_names:
    if chname == "AU_Detection_Challenge":
        raise NotImplementedError
        # TODO: Not finished
        cols_names = [
            "VideoID",
            "FrameID",
            "AU1",
            "AU2",
            "AU4",
            "AU6",
            "AU7",
            "AU10",
            "AU12",
            "AU15",
            "AU23",
            "AU24",
            "AU25",
            "AU26",
        ]
        path_to_load = os.path.join(*[PATH, chname, "Train_Set", "*.txt"])
        all_paths = glob(path_to_load)
        anno_data = []

        # For each video
        for path in tqdm(all_paths):
            lines = read_txt(path)

            # cols = lines[0]
            data = lines[1:]
            video_id = path.split("/")[-1].replace(".txt", "")

            all_frame_paths = glob(os.path.join(*[CROP_PATHS, video_id, "*.jpg"]))
            all_frame_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
            all_frame_paths = [fpath.split("/")[-1] for fpath in all_frame_paths]
            video_ids = [video_id] * len(all_frame_paths)
            # print(video_id, all_frame_paths[0:10], path, len(all_frame_paths), len(data))
            try:
                assert len(data) == len(all_frame_paths)
            except:
                fails.setdefault(chname, []).append(path)
                continue

            success.setdefault(chname, []).append(path)
            for vid, fid, d in zip(video_ids, all_frame_paths, data):
                _d = [vid, fid]
                _d.extend(d)
                anno_data.append(_d)

        df = pd.DataFrame(anno_data, columns=cols_names)
        break

    elif chname == "EXPR_Classification_Challenge":
        # TODO: Not finished
        raise NotImplementedError
        cols_names = [
            "VideoID",
            "FrameID",
            "Neutral",
            "Anger",
            "Disgust",
            "Fear",
            "Happiness",
            "Sadness",
            "Surprise",
            "Other",
        ]
        path_to_load = os.path.join(*[PATH, chname, "Train_Set", "*.txt"])
        all_paths = glob(path_to_load)

    elif chname == "VA_Estimation_Challenge":
        # TODO: Not finished
        raise NotImplementedError

    elif chname == "MTL_Challenge":
        print("> Convert MTL_Challenge to CSV file")
        path_to_load = os.path.join(*[PATH, chname, "*.txt"])
        cols_names = [
            "VideoID",
            "FrameID",
            "Valence",
            "Arousal",
            "Expression",
            "AU1",
            "AU2",
            "AU4",
            "AU6",
            "AU7",
            "AU10",
            "AU12",
            "AU15",
            "AU23",
            "AU24",
            "AU25",
            "AU26",
        ]
        all_paths = glob(path_to_load)  # Train split
        assert len(all_paths) == 2
        for path in all_paths:
            split = path.split("/")[-1].split("_")[0]
            lines = read_txt(path)
            # cols = lines[0]
            data = lines[1:]

            video_ids = [d[0].split("/")[0] for d in data]
            frame_ids = [d[0].split("/")[1] for d in data]

            assert len(video_ids) == len(data)
            anno_data = []
            missing_paths = []
            for vid, fid, d in zip(video_ids, frame_ids, data):
                img_path = os.path.join(*[CROP_PATHS, vid, fid])

                if not os.path.isfile(img_path):
                    missing_paths.append(img_path)
                    continue

                _d = [vid, fid]
                _d.extend(d[1:])
                anno_data.append(_d)
            print(
                f"{split.upper()}: Cannot find {len(missing_paths)} paths ({len(missing_paths)/len(video_ids)})"
            )
            df = pd.DataFrame(anno_data, columns=cols_names)
            df.to_csv(f"/mnt/DATA2/congvm/Affwild2/mtl_{split}_anno.csv", index=False)
