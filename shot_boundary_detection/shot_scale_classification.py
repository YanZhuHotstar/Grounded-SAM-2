'''
Author: Youqiang Zhang 
'''
import torch, torchvision
from collections import OrderedDict
import glob
import os
from PIL import Image
from torchvision import datasets, transforms
from collections import Counter
import argparse
import numpy as np
import statistics
import random 


def predict_one_shot(img_list):
    with torch.no_grad():
        tmp_pred = []
        outputs = []
        for img_path in img_list:
            img = Image.open(img_path)
            img = test_transform(img)
            img = img.to(device)
            img = img.unsqueeze(0)
            output = net(img)
            outputs.append(softmax(output))
            output = torch.argmax(output, dim=1).data.cpu().numpy()
            tmp_pred.append(output[0])
        tmp_pred = Counter(tmp_pred)
        pred = [tmp_pred[0], tmp_pred[1], tmp_pred[2]]
        cls = np.argmax(pred)
        outputs = torch.cat(outputs).cpu().numpy()
        prob = np.mean(outputs, axis=0)
        prob = [round(prob[0], 3), round(prob[1], 3), round(prob[2], 3)]
        # harmonic mean works better to average 'rates'; mean is better when list of numbers are in the same unit
        harmonic_mean = [-1,-1,-1] #[round(statistics.harmonic_mean(outputs[:, 0]), 3),
                                #round(statistics.harmonic_mean(outputs[:, 1]), 3),
                                #round(statistics.harmonic_mean(outputs[:, 2]), 3)]
    print("pred: ", pred, "prob: ", prob, sum(prob), 'harmonic_mean: ', harmonic_mean, sum(harmonic_mean))

    return cls, pred, prob, harmonic_mean

def infer_one_video(video, shot_dir, frame_path, max_frame=10):
    name = os.path.splitext(video)[0]
    shot_path = os.path.join(shot_dir, f'{name}_0.1.txt')
    result_path = os.path.join(shot_dir, f'{name}_scale.txt')
    # check if shot and frame paths exists 
    if not os.path.exists(shot_path):
        print('Shot detection result ', shot_path, 'not exists. Skipping current video ', video)
        return 
    # check if shot and frame paths exists
    if not os.path.exists(frame_path):
        print('Frame dir ', frame_path, 'not exists. Skipping current video ', video)
        return
    shots = np.loadtxt(shot_path, dtype=np.int32, ndmin=2)
    f = open(result_path, 'w')
    for id, (start, end) in enumerate(shots):
        frame_id_samples = random.sample(range(start+1, end+2), min(max_frame, end-start+1))
        img_list = sorted([os.path.join(frame_path, "{:08d}.png".format(x)) for x in frame_id_samples])
        r = predict_one_shot(img_list)
        f.write('{},{},{},{}\n'.format(*r))

    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract shot keyframes')
    parser.add_argument('--video_dir', help='path to video folder or video')
    parser.add_argument('--shot_dir', help='path to shot detection results')
    parser.add_argument('--frame_dir', help='path to frames')
    parser.add_argument('--check_point', help='path to model file', default='shot_scale.pth.tar')
    args = parser.parse_args()

    net = torchvision.models.mobilenet_v2(num_classes=3, width_mult=1.0, pretrained=False)
    state_dict = torch.load(args.check_point, map_location=torch.device('cpu'))['state_dict']
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    net.load_state_dict(new_state_dcit, strict=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    test_transform = transforms.Compose([
        transforms.Resize((270, 480)),
        transforms.CenterCrop((256, 456)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    softmax = torch.nn.Softmax()
    
    videos = [os.path.basename(args.video_dir)]
    if os.path.isdir(args.video_dir):
        videos = os.listdir(args.video_dir)
    for video in videos:
        infer_one_video(video, args.shot_dir, args.frame_dir)
