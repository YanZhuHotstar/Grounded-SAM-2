## Wiki 
Project page: 
[Shot Boundary Detection (SBD)](https://hotstar.atlassian.net/wiki/spaces/HP2/pages/2731706119/Shot+Boundary+Detection+SBD)

## Quick Start
* Tested only on Linux machine
```bash
# on ec2 instance, auto-download models from S3
bash setup.sh

# run inference
#     1. shot detection 
#     2. extract keyframes for both video tagging and scene detection
bash infer.sh $VIDEO_DIR
```

## Pre-trained model 
* [Google Drive Link](https://drive.google.com/drive/folders/10zNLZqrdd2fqymdO3Gx1MFW7KpWQIX4g?usp=sharing) 
* AWS S3 Link: ```s3://hotstar-ads-ml-us-east-1-prod/content-intelligence/video-tagging-models/transnetv2-weights.zip```

Unzip the model 

## Results 
Model | RAI | BBC Planet Earth | ClipShot | Hotstar
--- | :---: | :---: | :---: | :---:
 ffprobe| 83.92|89.62|23.21|90.27 
 TransNet| **94.17**|92.78|73.93|94.58
 TransNetV2| 92.91|**96.02**|**78.11**|**97.39**
 DeepSBD| 92.51|92.87|76.48|89.21

More detailed results are in Wiki page.

## Data 
1. Download RAI, BBC, ClipShot, Hotstar datasets from [here](https://drive.google.com/drive/folders/1GyhUTEirU1aQ6w6bXcwXNx7L-Cp9ulQy?usp=sharing) 
2. Use your own data - convention
Video name must match label name.
  - video
      - 1.mp4
      - 2.mp4
  - annotation 
      - 1.txt
      - 2.txt
    

## Inference
Higher value of --threshold generates less shots. 
```bash 
# run inference inside TransNetV2 directory
cd TransNetV2

python inference.py --result_dir hotstar_results --video ../data/HotstarDataset --threshold 0.1 --eval --gt_dir ../data/hotstar_gt --verbose

python inference.py --result_dir rai_results --video ../data/RaiShotDetection --threshold 0.5 --eval --gt_dir ../data/RaiShotDetection/annotation --verbose

python inference.py --result_dir bbc_results --video ../data/BBC_Planet_Earth_Dataset_downsampled --threshold 0.5 --eval --gt_dir ../data/BBC_Planet_Earth_Dataset/annotations/shots --verbose

python inference.py --result_dir clipshot_results --video ../data/ClipShots/ClipShots/videos --threshold 0.5 --eval --gt_dir ../data/ClipShots/ClipShots/annotations/test_gt
```

## Output file
Inside result directory, there could be three types of output associated with each video. 
1. ```<filename>_<threshold>.txt``` - Like Sanam_1000_0.1.txt. Contains a list of shots indicated with begin and end frame indices. Index starts at 0.  
2. ```<filename>_frame.txt``` - Like Sanam_1000_frame.txt. Each line is the probability of the corresponding frame being a transition frame. A threshold is appplied on these probablility scores to determine shot boundaries, which can be specified with "--threshold" argument.  
3. ```<filename>_<threshold>.png``` - Like Sanam_1000_0.1.png. An image with all frames plotted and transition boundary labeled with **light green** color.  

## Extract keyframes of every shot 
```bash
# For scene detection: 3 keyframes per shot
python extract_keyframe.py --video_dir $VIDEO_DIR --shot_dir $SHOT_DIR --keyf_dir $KEYF_DIR --mode scene --num_keyf 3
# For video tagging, 1 keyframe per second, minimum 3 keyframes per shot
python extract_keyframe.py --video_dir $VIDEO_DIR --shot_dir $SHOT_DIR --keyf_dir $KEYF_DIR --mode tagging --interval 1 --min_num_keyf 3
```

## References
* TransNetV2: An effective deep network architecture for fast shot transition detection [paper](https://arxiv.org/abs/2008.04838) ([source code](https://github.com/soCzech/TransNetV2))
* TransNet: A deep network for fast detection of common shot transitions [paper](https://arxiv.org/abs/1906.03363) ([source code](https://github.com/soCzech/TransNet))
* Large-scale, fast and accurate shot boundary detection through spatio-temporal convolutional neural networks [paper](https://arxiv.org/abs/1705.03281) ([source code](https://github.com/melgharib/DSBD))
* Fast Video Shot Transition Localization with Deep Structured Models [paper](https://arxiv.org/abs/1808.04234) ([source code](https://github.com/Tangshitao/ClipShots_basline))

