set -e
eval "$(conda shell.bash hook)"

NAME=shot_detection
if ! conda env list|grep $NAME; then
    conda create -y -n $NAME python=3.7
    conda activate $NAME
    pip install tensorflow-gpu==2.1.0 ffmpeg-python==0.2.0 pillow==8.2.0 tqdm protobuf==3.14.0 tensorflow==2.1.0 y4m
    conda install -y opencv=3.4.2 cudnn=7.6.5
fi

conda activate $NAME

#cd "$(dirname $0)"
#pip install --no-deps -e .

# download pretrained models
if ! ls|grep transnetv2-weights; then
    aws s3 cp s3://hotstar-ads-ml-us-east-1-prod/content-intelligence/video-tagging-models/transnetv2-weights.zip .
    unzip transnetv2-weights.zip
fi
