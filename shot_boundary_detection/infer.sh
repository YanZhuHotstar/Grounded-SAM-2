start_time=$(date +%s)

# calculate execution time
calculate_execution_time() {
  step_name=$1
  start_time=$2
  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "step -> '$step_name' execution time：$execution_time second(s)"
}

set -e
eval "$(conda shell.bash hook)"

NAME=shot_detection
conda activate $NAME

VIDEO_DIR=${1:-/home/ubuntu/data/test_video}
SHOT_DIR=shot_txt
KEYF_DIR=keyf
calculate_execution_time "prepare" $start_time

# Run Shot Detection on videos in $VIDEO_DIR
python inference.py --result_dir $SHOT_DIR --video $VIDEO_DIR --threshold 0.1
calculate_execution_time "Run Shot Detection on videos" $start_time

# Extract Keyframes
# For scene detection: 3 keyframes per shot
python extract_keyframe.py --video_dir $VIDEO_DIR --shot_dir $SHOT_DIR --keyf_dir ${KEYF_DIR}_scene --mode scene --num_keyf 3
calculate_execution_time "For scene detection: 3 keyframes per shot" $start_time
# For video tagging, 1 keyframe per second, minimum 3 keyframes per shot
python extract_keyframe.py --video_dir $VIDEO_DIR --shot_dir $SHOT_DIR --keyf_dir $KEYF_DIR --mode tagging --interval 1 --min_num_keyf 3
calculate_execution_time "For video tagging, 1 keyframe per second, minimum 3 keyframes per shot" $start_time
