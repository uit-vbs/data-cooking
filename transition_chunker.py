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
    transitions = df.to_dict('records')
    transitions.sort(key=lambda x: x['frame_start'])

    transitions_new = []
    current_transitions = []
    current_length = 0

    def add_transition(video_id, frame_start, frame_end):
        transitions_new.append({
            'video_id': video_id,
            'transition_id': f"{video_id}_T{len(transitions_new):06d}",
            'frame_start': frame_start,
            'frame_end': frame_end,
        })

    def collect_transition():
        nonlocal current_transitions
        nonlocal current_length
        video_id = current_transitions[0]['video_id']
        frame_start = current_transitions[0]['frame_start']
        num_transitions = current_length // MIN_FRAMES_PER_TRANSITION
        for i in range(num_transitions):
            add_transition(
                    video_id,
                    frame_start + current_length * i // num_transitions,
                    frame_start + current_length * (i+1) // num_transitions,
                )
        current_transitions = []
        current_length = 0

    for transition in transitions:
        transition_length = transition['frame_end'] - transition['frame_start'] + 1
        if current_length > 0 and current_length + transition_length > MIN_FRAMES_PER_TRANSITION:
            add_transition(
                current_transitions[0]['video_id'],
                current_transitions[0]['frame_start'],
                current_transitions[-1]['frame_end'],
            )
            current_transitions = []
            current_length = 0
        current_transitions.append(transition)
        current_length += transition_length
        if current_length > MIN_FRAMES_PER_TRANSITION:
            collect_transition()
    
    if current_length > MIN_FRAMES_PER_TRANSITION:
        collect_transition()

    pd.DataFrame(transitions_new).to_csv(os.path.join(output_dir, os.path.basename(file)), index=False)

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    files = glob.glob(os.path.join(input_dir, '*.csv'))
    files.sort()
    os.makedirs(output_dir, exist_ok=True)
    with Pool(32) as pool:
        results = list(tqdm.tqdm(pool.imap(process, files), total=len(files)))