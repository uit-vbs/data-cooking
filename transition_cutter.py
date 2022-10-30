import sys
import os
import glob
import logging

import cv2
import tqdm

import pandas as pd
from multiprocessing import Pool

logger = logging.getLogger(__name__)

def split_transitions(job):
    video_path = job['vid_path']
    output_path = job['out_path']
    df_transitions = pd.read_csv(job['trans_path'])

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w * 240 // h, 240)
    fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))

    rows = df_transitions.to_dict('records')
    rows.sort(key=lambda row: row['frame_start'])
    for row in rows:
        writer = cv2.VideoWriter(output_path % row['transition_id'], fourcc, fps, size, True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_start'])
        while(cap.isOpened() and writer.isOpened()):
            ret, frame = cap.read()
            if not ret: break
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if frame_number > row['frame_end']: break
            frame_resized = cv2.resize(frame, size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            writer.write(frame_resized)

        writer.release()
    cap.release()

ALLOWED_EXTENSIONS = ['.mp4']

if __name__ == '__main__':
    dir_vid = sys.argv[1]
    dir_out = sys.argv[2]
    dir_trans = sys.argv[3]
    os.makedirs(dir_out, exist_ok=True)

    list_vid = glob.glob(os.path.join(dir_vid, '*'))
    list_jobs = []
    for vid_path in list_vid:
        vid_file = os.path.basename(vid_path)
        vid_name, vid_ext = os.path.splitext(vid_file)
        if vid_ext.lower() not in ALLOWED_EXTENSIONS:
            logger.warn(f"Invalid extension for file {vid_path}")
            continue
        transition_file = glob.glob(os.path.join(dir_trans, f'*{vid_name}*.csv'))
        if len(transition_file) == 0:
            logger.warn(f"No matching transition_file for {vid_path}")
            continue
        if len(transition_file) > 1:
            logger.warn(f"More than one matching transition_file: {transition_file}")
            continue
        list_jobs.append({
            'vid_path': vid_path,
            'out_path': os.path.join(dir_out, "%s.mp4"),
            'trans_path': transition_file[0],
        })
    list_jobs.sort(key=lambda x: x['vid_path'])

    with Pool(4) as pool:
        results = list(tqdm.tqdm(pool.imap(split_transitions, list_jobs), total=len(list_jobs)))
