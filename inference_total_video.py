import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import time
from argparse import ArgumentParser
import glob
from tqdm import tqdm




def parse_args():
    parser = ArgumentParser(description='ROI Merge Inference Script')
    parser.add_argument('--frame_path', type=str, default='/efs/yanzhu/datas/grounded_sam2_work_dir/frames/BB7', help='Path to the frame images')
    parser.add_argument('--save_path', type=str, default='/efs/yanzhu/datas/grounded_sam2_work_dir/mask/BB7_ShotThr_0.45_person', help='Path to save the SAM2 mask results')
    parser.add_argument('--shot_bd_path', type=str, default='/efs/yanzhu/datas/grounded_sam2_work_dir/shot_boundary/BB7/BB7_Welcome_Song_HD_shortcut_crf28_0.45.txt', help='Path to the shot boundary file')
    parser.add_argument('--sam2_model_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml', help='Path to the model config file')
    parser.add_argument('--sam2_ckpt', type=str, default='./checkpoints/sam2.1_hiera_large.pt', help='Path to the model ckpt file')
    parser.add_argument('--model_id', type=str, default='IDEA-Research/grounding-dino-tiny', help='model_id of grounded')
    parser.add_argument('--prompt_text', type=str, default='person.', help='prompt_text of grounded')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model

    video_predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_ckpt)
    sam2_image_model = build_sam2(args.sam2_model_cfg, args.sam2_ckpt)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # init grounding dino model from huggingface
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

    shot_bd = []
    with open(args.shot_bd_path, 'r') as file:
        for line in file:
            start, end = map(int, line.strip().split('\t'))
            shot_bd.append((start, end))
    shot_bd = np.array(shot_bd)
    with torch.no_grad():
        for shot_idx in range(shot_bd.shape[0]):
            torch.cuda.empty_cache()
            inference_state = video_predictor.init_state(video_path=args.frame_path, start_idx=shot_bd[shot_idx, 0], end_idx=shot_bd[shot_idx, 1])
            video_predictor.reset_state(inference_state)
            img_path = os.path.join(args.frame_path, f"{str(shot_bd[shot_idx, 0]+1).zfill(8)}.jpg")
            image = Image.open(img_path)

            # run Grounding DINO on the image
            inputs = processor(images=image, text=args.prompt_text, return_tensors="pt").to(device)
            outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

            # prompt SAM image predictor to get the mask for the object
            image_predictor.set_image(np.array(image.convert("RGB")))

            # process the detection results
            input_boxes = results[0]["boxes"].cpu().numpy()
            OBJECTS = results[0]["labels"]
            # try:
            if len(input_boxes)>0:
                masks, scores, logits = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                # convert the mask shape to (n, H, W)
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                for mask_id in range(min(masks.shape[0],20)):
                    ann_mask = (masks[mask_id, :, :]*255).astype(np.uint8)
                    _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=mask_id+1,
                        mask=ann_mask,
                    )
                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for key, value in video_segments.items():
                    mask = np.zeros([image.size[1], image.size[0]], dtype=np.float32)
                    for instance_id, instance_mask in value.items():
                        # if results[0]['labels'][instance_id-1] != "person":
                        #     continue
                        instance_mask = instance_mask[0,:,:]
                        mask_temp = np.zeros([instance_mask.shape[0],instance_mask.shape[1]], dtype=np.float32)
                        mask_temp[instance_mask==True] = 1
                        mask += mask_temp
                    mask[mask>1] = 1
                    mask = mask*255
                    mask = mask.astype(np.uint8)
                    cv2.imwrite(os.path.join(args.save_path, f"{str(shot_bd[shot_idx, 0]+key).zfill(5)}.png"), mask)
            # except Exception as e:
            #     pass
    total_frames = len(glob.glob(os.path.join(args.frame_path, '*.jpg')))
    mask = np.zeros([1080, 1920], dtype=np.uint8)
    for i in tqdm(range(total_frames)):
        if not os.path.exists(os.path.join(args.save_path, f"{str(i).zfill(5)}.png")):
            print(i)
            cv2.imwrite(os.path.join(args.save_path, f"{str(i).zfill(5)}.png"), mask)
            
