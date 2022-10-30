import glob
import sys
import os
import tqdm

import pandas as pd
from multiprocessing import Pool

# Distribute transitions into small chunks.
#
# If a transition is too long, then split into smaller transitions.
# E.g.: a transition of 2-minutes long is split into 12 parts of 10 seconds each.
#
# If there are multiple consecutive short transitions, then merge them into one.
# E.g.: there are 4 transitions with length 2, 4, 3, 12 seconds,
#   then merge them into two chunks: 2+4+3=9 seconds, and 12 seconds.

MIN_FRAMES_PER_TRANSITION = 30 * 10 # about 10 seconds per transition for ~30fps

def process():
    df = pd.read_csv(input_path)
    df['video_name'] = df['video_id']
    df['channel_id'] = df['video_name'].apply(lambda x: x.split('_')[0][1:]).astype(int)
    df['video_id'] = df['video_name'].apply(lambda x: x.split('_')[1][1:]).astype(int)
    df['frame_id'] = df['feature_frame_index']
    df['keyframe_id'] = df['index']
    df = df[[
        'keyframe_id',
        'video_name',
        'channel_id',
        'video_id',
        'frame_id',
    ]]
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    process()