import numpy as np
import os
import argparse
from tqdm import tqdm
import json
import cv2

from transnet_utils import scenes_from_predictions, draw_video_with_scenes
from transnetv2 import TransNetV2

# setup GPU memory growth to true to avoid CUDNN_STATUS_INTERNAL_ERROR
import tensorflow as tf
for physical_device in tf.config.list_physical_devices('GPU'):
    print(physical_device)
    tf.config.experimental.set_memory_growth(physical_device, True)


VIDEO_FORMATS = ['mp4', 'm4v', 'mkv', 'webm', 'mov', 'avi', 'wmv', 'mpg', 'flv', 'mxf']

def valid_format(src_video):
    # check if src_video has video extension
    if src_video.split('.')[-1] not in VIDEO_FORMATS:
        print(f'WARNING: Kipping processing {src_video} due to invalid format. Should be within {VIDEO_FORMATS}.')
        return False
    return True

def shot2json(shots, movieid, fps=25, shot_fmt='transnetv2', json_path='shot.json'):
    j = {'task_type': 'shot detection', 'shot_fmt': shot_fmt}
    data = {}
    data[movieid] = {'fps': fps}
    for shotid, (start, end) in enumerate(shots):
        shotid = str(shotid)
        data[movieid][shotid] = {'start_frame': int(start), 'end_frame': int(end), 'start_time': start / fps, 'end_time': end / fps}
    j['data'] = data
    print('Converting text results to JSON format and saving to ', json_path)
    json.dump(j, open(json_path, 'w'))

def inference_one_video(src_video, args):
    # extract file name without extension and path
    name = os.path.splitext(os.path.basename(src_video))[0]
    result_file = os.path.join(args.result_dir, f'{name}_{args.threshold}.txt')
    result_json = os.path.join(args.result_dir, f'{name}.json')
    result_per_frame = os.path.join(args.result_dir, name + '_frame.txt')
    result_img = os.path.join(args.result_dir, f'{name}_{args.threshold}.png')

    # if frame prediction result exists, scene results can be generated directly
    # by thresholding frame predictions without re-inference the video
    if not os.path.exists(result_per_frame):
        print('INFO: Inferencing ', src_video)
        try:
            video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(src_video)
        except Exception as e:
            print(f'ERROR: Failed inferencing {src_video}; {e}')
            return
    else:
        print(f'INFO: Per-frame prediction result {result_per_frame} exists, reusing with threshold={args.threshold}.')
        single_frame_predictions = np.loadtxt(result_per_frame, dtype=float, ndmin=1)

    # Generate list of scenes from predictions, returns tuples of (start frame, end frame)
    scenes = scenes_from_predictions(single_frame_predictions, threshold=args.threshold)

    # write result file
    print('INFO: Saving shot transition results to ', result_file)
    with open(result_file, 'w') as f:
        for scene in scenes:
            f.write(f'{scene[0]}\t{scene[1]}\n')

    # save per-frame scores in order to fast tune args.THRESHOLD
    print('INFO: Saving per-frame prediction results to ', result_per_frame)
    with open(result_per_frame, 'w') as f:
        for score in single_frame_predictions:
            f.write(f'{score}\n')

    print('INFO: Saving shot transition results to ', result_json)
    cap = cv2.VideoCapture(src_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    shot2json(scenes, name, fps, json_path=result_json)

    # VISUALIZATION
    if args.visualization:
        print('INFO: Plotting shot transition boundaries...')
        img = draw_video_with_scenes(scenes=scenes, video_path=src_video)
        img.save(result_img)
        print(f'INFO: Image saved at {result_img}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run shot transition detection on videos in a folder or on a single video; "
                    "Option to visualize frame images with transition labels; Option to evaluate F1 score given ground truth folder.")
    # group args, grouping only impact help print, nothing else
    inference_group = parser.add_argument_group("Inference Arguments")
    eval_group = parser.add_argument_group("Evaluation Arguments")

    inference_group.add_argument("--result_dir", required=True, help="Path to save prediction results")
    inference_group.add_argument("--video", required=True, help="Path to a single video or a folder of videos")
    inference_group.add_argument("--model_dir", default=os.path.join(os.path.dirname(__file__), "transnetv2-weights/"),
                        help="Path to trained model, default is ./transnetv2-weights")
    inference_group.add_argument("--threshold", type=float, default=0.1, help="Threshold to be counted as transition frame, default is 0.1")
    inference_group.add_argument("--visualization", action='store_true', help="Draw frames with transitions labels")

    eval_group.add_argument("--eval", action='store_true', help="Enable evaluation.")
    eval_group.add_argument("--gt_dir", default=None, help="Path to ground truth to evaluate F1 score")
    eval_group.add_argument("--n_frames_miss_tolerance", type=int, default=2,
                        help="Max number of frame shifting tolerance to be counted as true positive")
    eval_group.add_argument("--verbose", action='store_true', help="Enable printing evaluation score of every file")
    args = parser.parse_args()

    # create result folder
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # location of learned weights is automatically inferred
    print('INFO: Loading model from ', args.model_dir)
    model = TransNetV2(model_dir=args.model_dir)
    print('INFO: Model loaded')

    # check if args.video is a file or folder, handle separately
    if os.path.isfile(args.video) and valid_format(args.video):
        inference_one_video(args.video, args)
    elif os.path.isdir(args.video):
        video_list = [os.path.join(args.video, x) for x in os.listdir(args.video)] # if os.path.isfile(os.path.join(args.video, x))]
        for src_video in tqdm(video_list):
            if valid_format(src_video):
                inference_one_video(src_video, args)
            elif os.path.isdir(src_video): 
                inference_one_video(src_video, args)
    else:
        print(f'ERROR: {args.video} must be either a video path or a video folder path.')
        exit()

    # Evaluation - F1 score
    if args.eval:
        if args.gt_dir is None:
            print('ERROR: --gt_dir must be specified to run evaluation.')
        else:
            from evaluate import evaluate
            evaluate(result_dir=args.result_dir, gt_dir=args.gt_dir,
                     n_frames_miss_tolerance=args.n_frames_miss_tolerance, verbose=args.verbose)


'''
### CLIPSHOT
100%|█████████████████████████████████████████████████████████| 490/490 [00:00<00:00, 2993.45it/s]
Overall - p=0.7439, r=0.7952, f1=0.7686, tp=5384, fp=1854, fn=1387


### BBC
bbc_09_0.5.txt  - p=0.9862, r=0.9808, f1=0.9835, tp=358, fp=5, fn=7
bbc_04_0.5.txt  - p=0.9890, r=0.9554, f1=0.9719, tp=450, fp=5, fn=21
bbc_05_0.5.txt  - p=0.9934, r=0.9804, f1=0.9868, tp=450, fp=3, fn=9
bbc_07_0.5.txt  - p=0.9824, r=0.9472, f1=0.9645, tp=502, fp=9, fn=28
bbc_03_0.5.txt  - p=0.9950, r=0.9500, f1=0.9720, tp=399, fp=2, fn=21
bbc_11_0.5.txt  - p=0.9842, r=0.9335, f1=0.9581, tp=435, fp=7, fn=31
bbc_06_0.5.txt  - p=0.9902, r=0.9657, f1=0.9778, tp=507, fp=5, fn=18
bbc_02_0.5.txt  - p=0.9890, r=0.9372, f1=0.9624, tp=358, fp=4, fn=24
bbc_08_0.5.txt  - p=0.9880, r=0.8020, f1=0.8853, tp=328, fp=4, fn=81
bbc_10_0.5.txt  - p=0.9815, r=0.8525, f1=0.9125, tp=318, fp=6, fn=55
bbc_01_0.5.txt  - p=0.9859, r=0.9459, f1=0.9655, tp=420, fp=6, fn=24
100%|████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 281.79it/s]
Overall - p=0.9878, r=0.9341, f1=0.9602, tp=4525, fp=56, fn=319


### RAI 
23558_0.5.txt  - p=0.9902, r=0.9712, f1=0.9806, tp=101, fp=1, fn=3
25009_0.5.txt  - p=0.9905, r=0.9541, f1=0.9720, tp=104, fp=1, fn=5
21867_0.5.txt  - p=0.9771, r=0.8767, f1=0.9242, tp=128, fp=3, fn=18
23557_0.5.txt  - p=0.9661, r=0.9500, f1=0.9580, tp=57, fp=2, fn=3
21829_0.5.txt  - p=0.9500, r=0.7125, f1=0.8143, tp=57, fp=3, fn=23
25010_0.5.txt  - p=0.9681, r=0.9286, f1=0.9479, tp=182, fp=6, fn=14
25011_0.5.txt  - p=0.8060, r=0.8852, f1=0.8438, tp=54, fp=13, fn=7
25012_0.5.txt  - p=0.8710, r=0.8571, f1=0.8640, tp=54, fp=8, fn=9
23553_0.5.txt  - p=0.9905, r=0.9286, f1=0.9585, tp=104, fp=1, fn=8
25008_0.5.txt  - p=0.9259, r=0.9259, f1=0.9259, tp=50, fp=4, fn=4
100%|██████████████████████████████████████████| 10/10 [00:00<00:00, 942.67it/s]
Overall - p=0.9550, r=0.9046, f1=0.9291, tp=891, fp=42, fn=94

### HOTSTAR 
Prem_Ratan_Dhan_Payo_1000_0.1.txt  - p=0.9844, r=1.0000, f1=0.9921, tp=63, fp=1, fn=0
Baahubali_1000_0.1.txt  - p=1.0000, r=0.9726, f1=0.9861, tp=71, fp=0, fn=2
YRKKH_2800_5min_0.1.txt  - p=0.8598, r=0.9583, f1=0.9064, tp=92, fp=15, fn=4
DIL_BECHARA_TEST_01_0.1.txt  - p=0.9474, r=0.9474, f1=0.9474, tp=18, fp=1, fn=1
Aravindhante_2000_0.1.txt  - p=0.9595, r=1.0000, f1=0.9793, tp=71, fp=3, fn=0
M_S_DHONI_1000_0.1.txt  - p=0.9886, r=0.9886, f1=0.9886, tp=87, fp=1, fn=1
Dishoom_1000_0.1.txt  - p=1.0000, r=0.9918, f1=0.9959, tp=121, fp=0, fn=1
DRISHYAM_1000_0.1.txt  - p=0.9700, r=0.9898, f1=0.9798, tp=97, fp=3, fn=1
Aravindhante_3000_0.1.txt  - p=0.9776, r=1.0000, f1=0.9887, tp=131, fp=3, fn=0
Aravindhante_1000_0.1.txt  - p=1.0000, r=1.0000, f1=1.0000, tp=108, fp=0, fn=0
THE_OUTSIDER_0.1.txt  - p=1.0000, r=0.9589, f1=0.9790, tp=70, fp=0, fn=3
Baahubali_2000_0.1.txt  - p=0.9754, r=1.0000, f1=0.9876, tp=119, fp=3, fn=0
MODFAM_10_19_clip_0.1.txt  - p=0.9474, r=0.9863, f1=0.9664, tp=72, fp=4, fn=1
CHEKKA_1000_0.1.txt  - p=1.0000, r=1.0000, f1=1.0000, tp=54, fp=0, fn=0
GPVIPL19T_clip_0.1.txt  - p=0.8305, r=1.0000, f1=0.9074, tp=98, fp=20, fn=0
Sanam_1000_0.1.txt  - p=0.9840, r=0.9919, f1=0.9880, tp=123, fp=2, fn=1
HateStory_1000_0.1.txt  - p=0.9508, r=0.9355, f1=0.9431, tp=58, fp=3, fn=4
100%|███████████████████████████████████████████████████████████| 17/17 [00:00<00:00, 1037.38it/s]
Overall - p=0.9610, r=0.9871, f1=0.9739, tp=1453, fp=59, fn=19

'''
