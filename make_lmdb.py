import glob
import sys
import os
import tqdm
import lmdb
import logging
import shutil
from multiprocessing import Pool

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig()

metadata_path = sys.argv[1]
features_dir = sys.argv[2]
out_dir = sys.argv[3]
shutil.rmtree(out_dir)

df_metadata = pd.read_csv(metadata_path)

features_files = glob.glob(os.path.join(features_dir, '*.npz'))
job_list = []

def process(job):
    file_path = job['file_path']
    frames = job['frames']
    features = list(np.load(file_path).get('feature_lst')[:, 0, 0, :])
    if features.shape[0] != len(frames):
        logger.warning(f"Inconsistent features count ({features.shape[0]} != {len(frames)}): {file_path}")
    env = lmdb.open(out_dir, map_size=10**11)
    with env.begin(write=True) as txn:
        for ind, feature in zip(frames, features):
            ind = ind.to_bytes(32, 'big')
            feature = feature.tobytes()
            txn.put(ind, feature)

if __name__ == '__main__':
    logger.info("Preparing job list")
    for file in tqdm.tqdm(features_files):
        file_name = os.path.basename(file)
        video_id = os.path.splitext(file_name)[0]
        frames = df_metadata[df_metadata['video_id'] == video_id]['index'].to_list()
        job_list.append({
            'file_path': file,
            'frames': frames,
        })
    logger.info(f"Prepared {len(job_list)} jobs")

    with Pool(32) as pool:
        results = list(tqdm.tqdm(pool.imap(process, job_list), total=len(job_list)))