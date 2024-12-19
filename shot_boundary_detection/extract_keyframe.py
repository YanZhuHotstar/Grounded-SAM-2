'''
Extract N keyframes from each shot given input video and shot detection results.
'''
import json

import cv2
import os
import glob
from tqdm import tqdm
import numpy as np
import argparse
import time
import subprocess
from functools import partial
from multiprocessing import Pool
import re


def save_keyf(cap, frame_indices, keyf_dir, video_basename, scale=0, mode='tagging'):
    '''
    save key frames of a video, naming convention assumes 1 key frame per shot;
    need to modify code if more than one keyf per shot
    '''
    save_dir = os.path.join(keyf_dir, video_basename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # read and save key frames
    for shotid, shot_idx in enumerate(frame_indices):
        for imgid, idx in enumerate(shot_idx):
            if idx > total_frame:
                print('WARNING: frame index {} larger than total number of frame.')
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if scale:
                w, h, c = frame.shape
                frame = cv2.resize(frame, (int(scale * h), int(scale * w)))
            if mode == 'tagging':
                # Video tagging naming convention
                img_name = '{}_{}_{}.jpg'.format(video_basename, shotid, imgid)
            else:
                # MovieNet naming convention
                img_name = 'shot_{:04d}_img_{}.jpg'.format(shotid, imgid)
            cv2.imwrite(os.path.join(save_dir, img_name), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 50])


def keyf2json(frame_indices, movieid, shot_dir):
    json_paths = [x for x in glob.glob(os.path.join(shot_dir, '*.json'))
                  if os.path.basename(x).startswith(movieid)]
    json_path = json_paths[0]
    print('Saving json output of keyframe extraction to ', json_path)
    j = json.load(open(json_path, 'r'))
    fps = j['data'][movieid]['fps']
    for shotid, shot_idx in enumerate(frame_indices):
        shotid = str(shotid)
        for imgid, frame in enumerate(shot_idx):
            j['data'][movieid][shotid][imgid] = {'frame': int(frame), 'timestamp': frame/fps}
    json.dump(j, open(json_path, 'w'))


def split_video_ffmpeg(input_video_path, shot_list, output_dir,
                       arg_override='-crf 21',
                       hide_progress=False, suppress_output=False):
    """
    Calls the ffmpeg command on the input video(s), generating a new video for
       each shot based on the start/end timecodes.
       type: (List[str], List[Tuple[FrameTimecode, FrameTimecode]], Optional[str], Optional[str], Optional[bool]) -> None
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    arg_override = arg_override.replace('\\"', '"')

    ret_val = None
    arg_override = arg_override.split(' ')
    try:
        processing_start_time = time.time()
        for i, (start_time, end_time) in enumerate(shot_list):
            duration = (end_time - start_time)
            # an alternative way to do it
            # duration = (end_time.get_frames()-1)/end_time.framerate - (start_time.get_frames())/start_time.framerate
            # duration_frame = end_time.get_frames()-1 - start_time.get_frames()
            call_list = ['ffmpeg']
            if suppress_output:
                call_list += ['-v', 'quiet']
            elif i > 0:
                # Only show ffmpeg output for the first call, which will display any
                # errors if it fails, and then break the loop. We only show error messages
                # for the remaining calls.
                call_list += ['-v', 'error']
            call_list += [
                '-y',
                '-ss',
                str(start_time),
                '-i',
                input_video_path]
            call_list += arg_override  # compress
            call_list += ['-map_chapters', '-1']  # remove meta stream
            call_list += [
                '-strict',
                '-2',
                '-t',
                str(duration),
                '-sn',
                os.path.join(output_dir, 'shot_{:04d}.mp4'.format(i))
                ]
            ret_val = subprocess.call(call_list)
            if not suppress_output and i == 0 and len(shot_list) > 1:
                print(
                    'Output from ffmpeg for shot 1 shown above, splitting remaining shots...')
            if ret_val != 0:
                break

    except OSError:
        print('ffmpeg could not be found on the system.'
                      ' Please install ffmpeg to enable video output support.')
    if ret_val is not None and ret_val != 0:
        print('Error splitting video (ffmpeg returned %d).', ret_val)


def get_keyf_indices(shots, num_keyf):
    frame_indices = []
    for id, (start, end) in enumerate(shots):
        _indices = []
        _indices.append(start)
        if num_keyf - 2 > 0:
            step = (end - start) / (num_keyf - 1)
            for _ in range(num_keyf - 2):
                _indices.append(min(end, int(start + step)))
                start += step
        _indices.append(end)
        frame_indices.append(sorted(set(_indices)))
    return frame_indices


def get_keyf_indices_by_time(shots, interval=1, min_num_keyf=3, fps=25):
    '''
    interval: sample 1 keyframe per interval seconds, 1 frame/second by default
    min_num_keyf: minimum number of keyframe per shot, minimum 3
    '''
    frame_indices = []
    for id, (start, end) in enumerate(shots):
        num_keyf = max(min_num_keyf, int((end - start + 1) / fps / interval) + 1)
        _indices = []
        if num_keyf > 1:
            step = (end - start) / (num_keyf - 1)
            _indices.append(start)
            while start < end:
                _indices.append(min(end, int(start + step)))
                start += step
            _indices.append(end)
        else:
            # num_keyf == 1, append middle frame
            _indices.append(int((start + end)/2))
        frame_indices.append(sorted(set(_indices)))
    return frame_indices


def process_one_video(video_path, shot_dir, keyf_dir, shot_video_dir=None, scale=1, num_keyf=3, interval=1, min_num_keyf=3, mode='tagging'):
    print('Processing ', video_path)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    # shot prediction results could have suffix, need to match video name; 'xxx_frame.txt' file contains only per frame sscores
    pattern = re.compile(video_basename + '_\d*\.\d*\.txt')
    shot_file_names = [x for x in os.listdir(shot_dir) if pattern.match(x)]
    if len(shot_file_names) == 0:
        print(f'ERROR: no shot prediction file for video {video_path}, skipping current video.')
    shot_path = os.path.join(shot_dir, shot_file_names[0])

    # read shot labels and compute keyframe indices
    shots = np.loadtxt(shot_path, dtype=np.int32, ndmin=2)
    # ffmpeg vsync option adds frames to the front of asynced video, causing frame index misalignment with opencv video cap
    # find frame offset
    cap = cv2.VideoCapture(video_path)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    offset = total_frame - (shots[-1][1] + 1)  # shot index starts at 0

    if offset != 0:
        print('WARNING: Frame number misalign between shot txt and video, this could be due to ffmpeg re-encoding. '
              'Changing shot frame indices to match with video. ')
        shots = [[max(0, s + offset), min(total_frame-1, e + offset)] for s, e in shots]

    # compute keyframe indices
    if mode == 'tagging':
        frame_indices = get_keyf_indices_by_time(shots, interval=interval, min_num_keyf=min_num_keyf, fps=cap.get(cv2.CAP_PROP_FPS))
    elif mode == 'scene':
        frame_indices = get_keyf_indices(shots, num_keyf)

    # save metadata to json
    keyf2json(frame_indices, video_basename, shot_dir)

    # read and save key frames from video
    save_keyf(cap, frame_indices, keyf_dir, video_basename, scale=scale, mode=mode)

    # split video into shots
    if shot_video_dir is not None:
        shots_time = shots / cap.get(cv2.CAP_PROP_FPS)
        split_video_ffmpeg(input_video_path=video_path, shot_list=shots_time,
                           output_dir=os.path.join(shot_video_dir, video_basename),
                           arg_override='-crf 21',
                           hide_progress=False, suppress_output=False)


def consistency_check(video, shot_dir, keyf_dir, num_keyf):
    '''
    Check if number of keyframes consistent with number of shot * n_keyf.
    Return video names with wrong number of keyframes.
    '''
    video_basename = os.path.splitext(video)[0]
    # Read shot detection results
    shot_file_names = [x for x in os.listdir(args.shot_dir) if x.startswith(video_basename) and x.endswith('.txt') and (not x.endswith('_frame.txt'))]
    if len(shot_file_names) == 0:
        print(f'ERROR: no shot prediction file for video {video}, skipping current video.')
        return
    shot_path = os.path.join(shot_dir, shot_file_names[0])
    num_shots = np.loadtxt(shot_path, dtype=np.int32, ndmin=2).shape[0]
    keyf_path = os.path.join(keyf_dir, video_basename)
    if not os.path.exists(keyf_path):
        print(f'{video} keyframe folder not exists.')
        return
    num_keyfs = len(os.listdir(keyf_path))
    if num_keyfs != num_shots * num_keyf:
        print(f'{video} number of keyframes inconsistent with number of shots.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract shot keyframes')
    parser.add_argument('--video_dir', help='path to video folder or video')
    parser.add_argument('--shot_dir', help='path to shot detection results')
    parser.add_argument('--keyf_dir', help='path to save shot keyframes')
    parser.add_argument('--shot_video_dir', help='path to save shot keyframes')
    parser.add_argument('--scale', type=int, default=0, help='scale to downsample keyframes, default is 1')
    parser.add_argument('--num_keyf', type=int, default=3, help='number of keyframes to extract per shot, default is 3')
    parser.add_argument('--parallel', type=int, default=1, help='process videos in paralell, 8 workers')
    parser.add_argument('--mode', type=str, default='tagging', choices=['tagging', 'scene'], help='extract keyframes for video tagging or scene detection')
    parser.add_argument('--interval', type=int, default=1, help='for video tagging, number of seconds to extract every keyframe in a shot')
    parser.add_argument('--min_num_keyf', type=int, default=3, help='for video tagging, minimum number of keyframes to extract per shot')
    args = parser.parse_args()
    '''
    python extract_keyframe.py --video_dir ../data/test_video --shot_dir hotstar_results --keyf_dir test_keyf --mode scene --num_keyf 3
    python extract_keyframe.py --video_dir ../data/test_video --shot_dir hotstar_results --keyf_dir test_keyf --mode tagging --interval 1
    '''
    # create output folder
    for output_dir in [args.keyf_dir, args.shot_video_dir]:
        if (output_dir is not None) and (not os.path.exists(output_dir)):
            os.makedirs(output_dir)

    if os.path.isdir(args.video_dir):
        video_files = glob.glob(os.path.join(args.video_dir, '*'))
    else:
        video_files = [args.video_dir]

    if args.parallel:
        print('--working in parallel')
        pool = Pool(8)
        _process_one_video = partial(process_one_video, shot_dir=args.shot_dir,
                                     keyf_dir=args.keyf_dir, shot_video_dir=args.shot_video_dir,
                                     scale=args.scale, num_keyf=args.num_keyf, interval=args.interval,
                                     min_num_keyf=args.min_num_keyf, mode=args.mode)
        pool.map(_process_one_video, video_files)
        pool.close()
        pool.join()

        # consistency check only in scene mode
        # if args.mode == 'scene':
            # pool = Pool(8)
            # _consistency_check = partial(consistency_check, shot_dir=args.shot_dir,
            #                              keyf_dir=args.keyf_dir, num_keyf=args.num_keyf)
            # pool.map(_consistency_check, video_files)
            # pool.close()
            # pool.join()

    else:
        print('--working not in parallel')
        for video in tqdm(video_files):
            process_one_video(video_path=video,
                              shot_dir=args.shot_dir,
                              keyf_dir=args.keyf_dir,
                              shot_video_dir=args.shot_video_dir,
                              scale=args.scale,
                              num_keyf=args.num_keyf,
                              interval=args.interval,
                              min_num_keyf=args.min_num_keyf,
                              mode=args.mode)





