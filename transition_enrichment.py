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

def process(file):
    df = pd.read_csv(file)
    df['transition_name'] = df['transition_id']
    df['channel_id'] = df['transition_name'].apply(lambda x: x.split('_')[0][1:]).astype(int)
    df['video_id'] = df['transition_name'].apply(lambda x: x.split('_')[1][1:]).astype(int)
    df['transition_id'] = df['transition_name'].apply(lambda x: x.split('_')[2][1:]).astype(int)
    df = df[[
        'channel_id',
        'video_id',
        'transition_id',
        'transition_name',
        'frame_start',
        'frame_end',
    ]]
    df.to_csv(os.path.join(output_dir, os.path.basename(file)), index=False)

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    files = glob.glob(os.path.join(input_dir, '*.csv'))
    files.sort()
    os.makedirs(output_dir, exist_ok=True)
    with Pool(32) as pool:
        results = list(tqdm.tqdm(pool.imap(process, files), total=len(files)))