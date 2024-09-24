from PIL.ImageOps import colorize, scale
import gradio as gr
import cv2
from PIL import Image
import zipfile
import torch
import time, math
import numpy as np
import os,io
import copy
from copy import  deepcopy
from glob import glob
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# from skimage.morphology.binary import binary_dilation
import matplotlib.pyplot as plt
import ffmpeg
import shutil
from tqdm import tqdm
import math
import time

from concurrent.futures import ThreadPoolExecutor, as_completed


### MISC
def remove_edge_cells(mask_image):
    w,h = mask_image.shape
    pruned_mask = copy.deepcopy(mask_image)
    remove_list = []
    edges = mask_image[0,:],mask_image[w-1,:],mask_image[:,0],mask_image[:,h-1]
    for edge in edges:
        edge_masks = np.unique(edge)
        for edge_mask in edge_masks:
            remove_list.append(edge_mask)
            pruned_mask[np.where(mask_image==edge_mask)] = 0

    return pruned_mask


def remove_small_cells(mask_image,area_threshold=10000):
    w,h = mask_image.shape
    pruned_mask = copy.deepcopy(mask_image)
    for mask_index in np.unique(mask_image):
        # if mask_index == mask_image[330,640]:
        #     a = 1
        area = np.sum(mask_image == mask_index)
        if area < area_threshold:
            pruned_mask[np.where(mask_image == mask_index)] = 0


    return pruned_mask




def split_cell_masks(mask_image):
    # Convert the mask image to grayscale
    gray_mask = cv2.cvtColor(np.array(mask_image*255,dtype=np.uint8), cv2.COLOR_BGR2GRAY)

    # Find unique RGB values in the mask image
    unique_colors = np.unique(mask_image.reshape(-1, mask_image.shape[2]), axis=0)

    # Create an empty index mask
    index_mask = np.zeros_like(gray_mask)

    # Assign unique index values to each cell in the index mask
    for i, color in enumerate(unique_colors):
        # Find pixels with the current color in the mask image
        color_mask = np.all(mask_image == color, axis=2)

        # Assign the index value to the corresponding pixels in the index mask
        index_mask[color_mask] = i + 1

    index_mask[np.where(gray_mask==255)]=0

    return index_mask


def remove_concentric_masks(mask_image):
    # Convert the mask image to grayscale
    cell_values = np.unique(mask_image)
    for i in range(1, len(cell_values)):# remove background
        mask_one = np.array(mask_image == cell_values[i],dtype=np.uint8)
        # mask_one_dilated = cv2.dilate(mask_one, np.ones((5, 5), np.uint8),100)
        # xmin, xmax, ymin, ymax = np.min(np.where(mask_one == 1)[0]), np.max(np.where(mask_one == 1)[0]),\
        #     np.min(np.where(mask_one == 1)[1]), np.max(np.where(mask_one == 1)[1]),
        contour, _ = cv2.findContours(mask_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contour) > 0:
            largest_contour = max(contour, key=cv2.contourArea)

            mask_image = cv2.drawContours(mask_image, [largest_contour], -1, (int(cell_values[i])), thickness=cv2.FILLED)

    return mask_image

def remove_debris(mask_image,circlular_threshold=0.5):
    # Convert the mask image to grayscale
    cell_values = np.unique(mask_image)
    for i in range(1, len(cell_values)):# remove background
        mask_one = np.array(mask_image == cell_values[i],dtype=np.uint8)

        # Calculate properties from the mask
        contour, _ = cv2.findContours(mask_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_idx = np.argmax([len(contour[i]) for i in range(len(contour))])

        perimeter = cv2.arcLength(contour[contour_idx], True)

        area = cv2.contourArea(contour[contour_idx])
        try:
            compactness = abs((perimeter ** 2) / (area * 4 * math.pi) - 1)
        except ZeroDivisionError:
            compactness = 10
            a = 1

        if compactness > circlular_threshold:
            mask_image[np.where(mask_image == cell_values[i])] = 0

    return mask_image




def dice_coefficient_multiclass(y_true, y_pred, num_classes):
    """
    Function to calculate the Dice Coefficient for multi-class segmentation.

    Parameters:
    y_true (np.array): Ground truth segmentation image with class labels.
    y_pred (np.array): Predicted segmentation image with class labels.
    num_classes (int): Number of classes in the segmentation.

    Returns:
    float: Average Dice coefficient across all classes.
    """
    dice_scores = np.zeros(num_classes)

    for class_id in range(num_classes):
        true_class = y_true == class_id
        pred_class = y_pred == class_id

        # Avoid division by zero
        if np.sum(true_class) + np.sum(pred_class) == 0:
            dice_scores[class_id] = 1
        else:
            intersection = np.logical_and(true_class, pred_class)
            dice_scores[class_id] = (2. * intersection.sum()) / (true_class.sum() + pred_class.sum())

    return np.mean(dice_scores)

### misc

def delete_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        
def clear_all(predictor, inference_state, old_video_segments,io_args, obj_counter):
    del predictor
    del inference_state
    del old_video_segments
    del io_args
    del obj_counter
    torch.cuda.empty_cache()
    # del obj_ids_dict

def get_meta_from_video(input_video, predictor, inference_state, old_video_segments, io_args, obj_counter, prgrs_bar=gr.Progress(track_tqdm=True)):
    if input_video is None:
        return None, None,""
    prgrs_bar(0, desc="Starting")
    
    # check if sam has been used before
    if predictor is not None:
        print('clearing all')
        clear_all(predictor, inference_state, old_video_segments, io_args, obj_counter)
    
    video_name = os.path.basename(input_video).split('.')[0]
    # create dir to save result 
    tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "tracking_results", f"{video_name}")}'
    io_args = {
        # 'file_name':video_name,
        'tracking_result_dir': tracking_result_dir,
        'video_frame': f'{tracking_result_dir}/outputs/frame',
        'output_mask_dir': f'{tracking_result_dir}/outputs/{video_name}_masks',
        'output_masked_frame_dir': f'{tracking_result_dir}/outputs/{video_name}_masked_frames',
        'output_mask_dir_png': f'{tracking_result_dir}/outputs/{video_name}_masks/png',
        # 'output_masked_frame_dir_png': f'{tracking_result_dir}/{video_name}_masked_frames/png',
        'output_mask_dir_np': f'{tracking_result_dir}/outputs/{video_name}_masks/numpy',
        # 'output_masked_frame_dir_np': f'{tracking_result_dir}/{video_name}_masked_frames/numpy',
        # 'output_video': f'{tracking_result_dir}/{video_name}_seg.mp4', # keep same format as input video
        # 'zip_mask': f'{tracking_result_dir}/{video_name}_pred_mask.zip'
    }
    
    # remove existing files
    delete_dir(io_args["tracking_result_dir"])
    
    #create directory and folder
    for key in io_args:   
        if not os.path.isdir(io_args[key]):
            os.makedirs(io_args[key])
            
    print("get meta information of input video")
    prgrs_bar(0.3, desc="Attempting to extract frames")
    vidcap = cv2.VideoCapture(input_video)
    # fps = vidcap.get(cv2.CAP_PROP_FPS)
    # frame_idx = 0
    
    # success, first_frame = vidcap.read()
    prgrs_bar(0.5, desc="Extracting frames")
    # curr_frame = first_frame
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    vidcap.release()
    # for frame_idx in prgrs_bar.tqdm(range(total_frames), desc="Extracting frames..."):
    frame_interval = 1

    ffmpeg.input(input_video).output(
        os.path.join(io_args["video_frame"], '%07d.jpg'), q=2, start_number=0, 
        vf=rf'select=not(mod(n\,{frame_interval}))', vsync='vfr'
    ).run()
    first_frame_path = os.path.join(io_args["video_frame"], '0000000.jpg')
    first_frame = cv2.imread(first_frame_path)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    # while success: #extract all the frames
    #     save_loc = os.path.join(io_args["video_frame"],"%05d.jpg" % frame_idx)
    #     cv2.imwrite(save_loc, curr_frame)
    #     print(f'extracting frame {frame_idx}')
    #     success, curr_frame = vidcap.read()
    #     frame_idx += 1
    

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    prgrs_bar(0.7, desc="Uploading frames to SAM2!")
    predictor, inference_state = init_sam2(io_args)
    prgrs_bar(1, desc="SAM2 initialised!")
    return first_frame_rgb, first_frame_rgb, io_args, predictor, inference_state, 1,0, dict(), dict()


def init_sam2(io_args):
    sam2_checkpoint = "./checkpoint/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    # sam2_checkpoint = "./checkpoint/sam2_hiera_base_plus.pt"
    # model_cfg = "sam2_hiera_b+.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=io_args['video_frame'], offload_state_to_cpu=True, async_loading_frames=True)
    predictor.reset_state(inference_state)
    
    # curr_obj_id = dict()
    print("SAM initialised!")
    return predictor, inference_state

def init_sam2_everything():
    sam2_checkpoint = "./checkpoint/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    # sam2_checkpoint = "./checkpoint/sam2_hiera_base_plus.pt"
    # model_cfg = "sam2_hiera_b+.yaml"
    
    sam2_everything = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_everything)
    return mask_generator



def add_new_obj(curr_obj_id, obj_counter):
    obj_counter += 1
    curr_obj_id = obj_counter
    return curr_obj_id, obj_counter

def draw_outline(mask, frame):
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    return frame

def labelled_mask(mask, image, obj_id):
   
    pos_x, pos_y = np.where(mask[0]) # H, W
    
    if not (np.isnan(np.mean(pos_x)) or np.isnan(np.mean(pos_y))): # check if mask disappeared
        center =int(np.mean(pos_y)),int(np.mean(pos_x)) # arrange according to cv2 standard
        # print(center)
        # Using cv2.putText() method
        image = cv2.putText(image, f'{obj_id}', center, cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0,0,0), 2, cv2.LINE_AA)
    return image
    

def show_mask(mask, image=None, obj_id=None):
    # cmap = plt.get_cmap("tab20")
    cmap_idx = 0 if obj_id is None else obj_id
    # color = np.array([*cmap(cmap_idx)[:3], 0.6])
    np.random.seed(cmap_idx)
    color = np.concatenate([np.random.random(3), [0.6]])

    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) # h, w, c (4 channels)
    mask_image = (mask_image * 255).astype(np.uint8)  # scale colour with 255
    # print(mask_image.shape)
    
    ## more elegant way to overlay image and mask (ref: medSAM)
    if image is not None:
        image_h, image_w = image.shape[:2]
        if (image_h, image_w) != (h, w):
            raise ValueError(f"Image dimensions ({image_h}, {image_w}) and mask dimensions ({h}, {w}) do not match")
        alpha_mask = mask_image[..., 3] / 255.0 #scale mask pixel value to normal , (h,w)
        alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=-1) # h,w,c
        image[..., :3] = np.where(alpha_mask > 0, (1 - alpha_mask) * image[..., :3] + alpha_mask * mask_image[..., :3], image[..., :3]) # assign value on pixel location that contain the mask
        return image
    return mask_image

    
def sam_click(predictor, origin_frame, inference_state, point_mode, obj_ids_dict, curr_obj_id, frame_num, obj_stack,evt:gr.SelectData):
    """
    Args:
        origin_frame: nd.array
        click_stack: [[coordinate], [point_mode]]
    """ 
    print("Click")
    point = np.array([[evt.index[0], evt.index[1]]], dtype=np.float32)
    # for mode, `1` means positive click and `0` means negative click
    if point_mode == "Positive":
        if curr_obj_id in obj_ids_dict.keys():
            obj_ids_dict[curr_obj_id]["coord"]= np.append(obj_ids_dict[curr_obj_id]["coord"], point, axis=0)
            # obj_ids_dict[curr_obj_id]["coord"].append(np.array([[evt.index[0], evt.index[1]]], dtype=np.float32))
            obj_ids_dict[curr_obj_id]["mode"]= np.append(obj_ids_dict[curr_obj_id]["mode"], np.array([1], np.int32), axis=0)
        else:
            obj_ids_dict[curr_obj_id] = {"coord": point, 
                                         "mode": np.array([1], np.int32)}
    else:
        if curr_obj_id in obj_ids_dict.keys():
            # obj_ids_dict[curr_obj_id]["coord"].append(np.array([[evt.index[0], evt.index[1]]], dtype=np.float32))
            obj_ids_dict[curr_obj_id]["coord"]= np.append(obj_ids_dict[curr_obj_id]["coord"], point, axis=0)
            obj_ids_dict[curr_obj_id]["mode"]= np.append(obj_ids_dict[curr_obj_id]["mode"], np.array([0], np.int32), axis=0)
        else:
            obj_ids_dict[curr_obj_id] = {"coord": point, 
                                         "mode": np.array([0], np.int32)}
    obj_stack.append(curr_obj_id)
    # ann_frame_idx = frame_num
    print([evt.index[0], evt.index[1]])
    print(frame_num)
    # add mask according to prompt
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_num,
        obj_id=curr_obj_id,
        points=obj_ids_dict[curr_obj_id]["coord"],
        labels=obj_ids_dict[curr_obj_id]['mode'],
    )
    print(curr_obj_id)
    masked_frame = origin_frame.copy()
    print(obj_ids_dict[curr_obj_id])
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy() # C,H,W
        masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
        masked_frame = labelled_mask(mask,masked_frame, obj_id)
    # masked_frame =cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
    # print(inference_state)
    return predictor, masked_frame, obj_ids_dict, obj_stack

def segment_everything(predictor, inference_state, origin_frame, prgrs_bar=gr.Progress(track_tqdm=True)):
    if inference_state is None:
        raise gr.Error("Please load video and 'Activate SAM' first!")
    prgrs_bar(0, desc="Starting segmentation")
    t1 = time.time()
    prgrs_bar(0, desc="Initialising SAM2 and segmenting image")
    mask_generator = init_sam2_everything()
    masks = mask_generator.generate(origin_frame)
    prgrs_bar(0.5, desc="Postprocessing the masks")
    ## remove background annotations
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    sorted_anns.pop(0)
    # img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    # img[:, :, 3] = 0
    # for ann in sorted_anns:
    #     m = ann['segmentation']

    #     if ann['area'] < 1:
    #         sorted_anns.pop(ann)
    #     # else:
    #     #     color_mask = np.concatenate([np.random.random(3), [0.35]])
    #     #     img[m] = color_mask
    # # merged_mask = split_cell_masks(img)
    # # merged_mask = remove_edge_cells(merged_mask)
    # # merged_mask = remove_concentric_masks(merged_mask)
    # # merged_mask = remove_debris(merged_mask,circlular_threshold=0.5)
    # print('Number of Instances Identified in the First Frame is %i'%len(sorted_anns))
    # # img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    # # img[:,:,3] = 0
    # # lowres_side_length = predictor.image_size // 4
    
    # dtype = next(predictor.parameters()).dtype
    # device = "cuda"
    # for mask_idx, mask_results in enumerate(sorted_anns):
    #     # Get mask into form expected by the model
    #     mask_tensor = torch.tensor(mask_results["segmentation"], dtype=dtype, device=device)
    #     # lowres_mask = torch.nn.functional.interpolate(
    #     #     mask_tensor.unsqueeze(0).unsqueeze(0),
    #     #     size=(lowres_side_length, lowres_side_length),
    #     #     mode="bilinear",
    #     #     align_corners=False,
    #     # ).squeeze()

    #     # Add each mask as it's own 'object' to segment
    #     _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
    #         inference_state=inference_state,
    #         frame_idx=0,
    #         obj_id=mask_idx+1, # add 1 to prevent start from 0
    #         mask=mask_tensor,
    #     )

            
    
        
    # masked_frame = origin_frame.copy()
    # # print(obj_ids_dict[curr_obj_id])
    # for i, obj_id in enumerate(out_obj_ids):
    #     mask = (out_mask_logits[i] > 0.0).cpu().numpy()
    #     masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
    #     masked_frame = labelled_mask(mask,masked_frame, obj_id)
    
    ## Xiaodan Method
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']

        if ann['area'] < 1:
            pass
        else:
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

    merged_mask = split_cell_masks(img)
    merged_mask = remove_edge_cells(merged_mask)
    merged_mask = remove_concentric_masks(merged_mask)
    merged_mask = remove_debris(merged_mask,circlular_threshold=0.5)

    merged_mask_correct = deepcopy(merged_mask)
    # correct the index for pred masks
    for idx,value in enumerate(list(np.unique(merged_mask))):
        merged_mask_correct[np.where(merged_mask==value)] = idx

    num_obj = idx
    print('Number of Instances Identified in the First Frame is %i'%num_obj )

    centers = []
    for i in range(1, num_obj+1):
        instance_mask = merged_mask_correct == i
        pos_x, pos_y = np.where(instance_mask)
        centers.append([int(np.mean(pos_y)),int(np.mean(pos_x))])

    print('Calculating the Centers, Completed')
    prgrs_bar(1, desc="Postprocessing complete")

    ann_frame_idx = 0  # the frame index we interact with


    for i in prgrs_bar.tqdm(range(num_obj), desc="Outputting masks to SAM2"):
        ann_obj_id = i + 1
        points = np.array([centers[i]])
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1], np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        
    masked_frame = origin_frame.copy()
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
        masked_frame = labelled_mask(mask,masked_frame, obj_id)
       
    
    del mask_generator
    t2= time.time()
    print(f"Time to segment everything:{t2-t1}")
    return predictor, masked_frame, len(out_obj_ids)
        
def process_frame(out_frame_idx, io_args, frame_files, new_video_segments, mask_output_paths, combined_output_paths):
    frame_path = os.path.join(io_args['video_frame'], frame_files[int(out_frame_idx)])
    image = cv2.imread(frame_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masked_frame = image.copy()

    mask_output = np.zeros_like(masked_frame)
    mask_np_output = np.zeros_like(masked_frame[..., 0])

    for obj_id, mask in new_video_segments[out_frame_idx].items():
        masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
        mask_output = show_mask(mask, image=mask_output, obj_id=obj_id)
        mask_np_output[mask[0] == True] = obj_id
        masked_frame = labelled_mask(mask, masked_frame, obj_id)

    mask_output = cv2.cvtColor(mask_output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mask_output_paths[out_frame_idx], mask_output)

    combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(combined_output_paths[out_frame_idx], combined_image_bgr)
    

def begin_track(predictor, inference_state, io_args, input_video, old_video_segments, frame_num, progress=gr.Progress(track_tqdm =True)):
    if inference_state is None:
        raise gr.Error("Please load video and 'Activate SAM' first!")
    if len(inference_state["point_inputs_per_obj"]) == 0:
        raise gr.Error("No points are provided; please add points first")
    
    t1 = time.time()
    temp_video_segments = copy.deepcopy(old_video_segments) # need to make copy to prevent same reference with new
    frame_files = sorted([f for f in os.listdir(io_args['video_frame']) if f.endswith('.jpg')])
    # run propagation throughout the video and collect the results in a dict
    if old_video_segments:
        new_video_segments = old_video_segments  # video_segments contains the per-frame segmentation results (frame id: {obj_id: (1,H,W)})
    else:
        new_video_segments ={}
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        new_video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    t2 = time.time()
    print(f"Time taken to track across video:{t2-t1}")
    
    video_name = os.path.basename(input_video).split('.')[0]
    out_path = f'{io_args["tracking_result_dir"]}/{video_name}_output.mp4'
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(out_path, fourcc, fps, (width, height)) 

    # ## add non-modified frame into video
    # if frame_num != 0: # check if adding new masks
    #     start_frame_idx = frame_num
    #     print(start_frame_idx)  
    #     for out_frame_idx in range(0, start_frame_idx):
    #         combined_output_path = os.path.join(io_args['output_masked_frame_dir'], f'{out_frame_idx:07d}.png')
    #     ## prepare to add mask
    #         combined_image_bgr = cv2.imread(combined_output_path)
    #         out.write(combined_image_bgr)
    
    t3 = time.time()
    # Precompute file paths outside the loop
    print("creating output files path")
    mask_output_paths = [os.path.join(io_args['output_mask_dir_png'], f'{i:07d}.png') for i in range(len(frame_files))]
    combined_output_paths = [os.path.join(io_args['output_masked_frame_dir'], f'{i:07d}.png') for i in range(len(frame_files))]
    
    # Using threading to extract mask faster (parallel computation)
    with ThreadPoolExecutor(max_workers=6) as executor:  # Adjust max_workers based on CPU capacity
        futures = [
            executor.submit(
                process_frame, 
                i, io_args, frame_files, new_video_segments, mask_output_paths, combined_output_paths
            )
            for i in range(len(frame_files))
        ]
        
            # Use tqdm to track the progress as tasks complete
        with tqdm(total=len(futures), desc="Outputting to video") as pbar:
            for future in as_completed(futures):
                # result = future.result()  # Get the result of the completed future
                pbar.update(1)  # Update the progress bar after each task completion
    # for out_frame_idx in progress.tqdm(range(len(frame_files)), desc='Outputting to Video'):
    #     # frame_path = os.path.join(io_args['video_frame'], frame_files[out_frame_idx])
    #     frame_path = frame_paths[out_frame_idx]
    #     ## prepare to add mask
    #     image = cv2.imread(frame_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     masked_frame = image.copy()
        
    #     ## disable feature for now
    #     # ## combine future mask addons
    #     # if temp_video_segments is not None and temp_video_segments.get(out_frame_idx):
    #     #     new_video_segments[out_frame_idx] = temp_video_segments[out_frame_idx] | new_video_segments[out_frame_idx]
        
    #     mask_output = np.zeros_like(masked_frame)
    #     mask_np_output = np.zeros_like(masked_frame[...,0])
    #     # print(mask_np_output.shape)
    #     ## add mask depending on num of obj in frame
    #     for obj_id, mask in new_video_segments[out_frame_idx].items():
    #             masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
    #             mask_output = show_mask(mask, image=mask_output, obj_id=obj_id)
    #             mask_np_output[mask[0]==True] = obj_id
    #             masked_frame = labelled_mask(mask,masked_frame, obj_id)
    #     # yield masked_frame, None, None, None
    #     ## save annotated (masked) image
    #     # mask_output_path = os.path.join(io_args['output_mask_dir_png'], f'{out_frame_idx:07d}.png')
    #     mask_output = cv2.cvtColor(mask_output, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(mask_output_paths[out_frame_idx], mask_output)
        
        
    #     # combined_output_path = os.path.join(io_args['output_masked_frame_dir'], f'{out_frame_idx:07d}.png')
    #     combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
    #     out.write(combined_image_bgr)
    #     cv2.imwrite(combined_output_paths[out_frame_idx], combined_image_bgr)

    # for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    #     ## get original frames
    #     # frame_idx = int(os.path.splitext(frame_files[out_frame_idx])[0])
    #     frame_path = os.path.join(io_args['video_frame'], frame_files[out_frame_idx])
    #     ## prepare to add mask
    #     image = cv2.imread(frame_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     masked_frame = image.copy()
    #     new_video_segments[out_frame_idx] = {
    #         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    #         for i, out_obj_id in enumerate(out_obj_ids)
    #     }
        
    #     ## combine future mask addons
    #     if temp_video_segments is not None and temp_video_segments.get(out_frame_idx):
    #         new_video_segments[out_frame_idx] = temp_video_segments[out_frame_idx] | new_video_segments[out_frame_idx]
        
    #     mask_output = np.zeros_like(masked_frame)
    #     mask_np_output = np.zeros_like(masked_frame[...,0])
    #     # print(mask_np_output.shape)
    #     ## add mask depending on num of obj in frame
    #     for obj_id, mask in new_video_segments[out_frame_idx].items():
    #             masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
    #             mask_output = show_mask(mask, image=mask_output, obj_id=obj_id)
    #             mask_np_output[mask[0]==True] = obj_id
    #             masked_frame = labelled_mask(mask,masked_frame, obj_id)
    #     # yield masked_frame, None, None, None
    #     ## save annotated (masked) image
    #     mask_output_path = os.path.join(io_args['output_mask_dir_png'], f'{out_frame_idx:07d}.png')
    #     mask_output = cv2.cvtColor(mask_output, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(mask_output_path, mask_output)
        
        
    #     combined_output_path = os.path.join(io_args['output_masked_frame_dir'], f'{out_frame_idx:07d}.png')
    #     combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
    #     # combined_image_bgr = masked_frame
        
        
    #     out.write(combined_image_bgr)
    #     cv2.imwrite(combined_output_path, combined_image_bgr)
    #     np_file = np.array(mask_np_output)
    #     np.save(f"{io_args['output_mask_dir_np']}/{out_frame_idx:07d}.npy", np_file)
    
    ffmpeg.input(os.path.join(io_args['output_masked_frame_dir'], '%07d.png'), framerate=fps).output(out_path, vcodec='libx264', pix_fmt='yuv420p').run()
    # out.release()
    
    ## zip files
    zip_path = f"{io_args['tracking_result_dir']}/{video_name}.zip"
    zip_folder(f"{io_args['tracking_result_dir']}/outputs", zip_path)

    # os.system(f"zip -r {zip_path} {io_args['output_mask_dir']}")
    print("done")
    t4 = time.time()
    print(f"time taken to export to video:{t4-t3}")
    combined_image_bgr = cv2.imread(combined_output_paths[-1])
    # combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
    
    return cv2.cvtColor(combined_image_bgr, cv2.COLOR_BGR2RGB), out_path, zip_path, new_video_segments

def show_res_by_slider(frame_percent, io_args):
    if io_args is None:
        raise gr.Error("Load video and do tracking first!")
    
    combined_output_path = io_args['output_masked_frame_dir']
    ## get directory of images
    combined_imgs_path = sorted(glob(combined_output_path+'/*.png'))
    
    total_frames = len(combined_imgs_path)
    
    chosen_frame_idx = math.floor(total_frames * frame_percent/100) # get frame based on percentage
    if frame_percent == 100:
        chosen_frame_idx = chosen_frame_idx -1 # comply with pythonic notation
    
    chosen_frame_path = combined_imgs_path[chosen_frame_idx]
    chosen_frame_img = cv2.imread(chosen_frame_path)
    chosen_frame_img = cv2.cvtColor(chosen_frame_img, cv2.COLOR_BGR2RGB) 
    
    return chosen_frame_img, f'Chosen Frame: {chosen_frame_idx+1}/{total_frames}', chosen_frame_idx

def send_to_board(io_args, frame_num, video_segments, predictor, inference_state, obj_ids_dict):
    if video_segments is None:
        raise gr.Error("Please track items in video before refining any objects!")
    combined_output_path = io_args['output_masked_frame_dir']
    
    ## get directory of images
    chosen_frame_path = sorted(glob(combined_output_path+'/*.png'))[frame_num]
    print(f"chosen frame: {chosen_frame_path}")
    chosen_frame_img = cv2.imread(chosen_frame_path)
    chosen_frame_img = cv2.cvtColor(chosen_frame_img, cv2.COLOR_BGR2RGB)
    frame_files = sorted([f for f in os.listdir(io_args['video_frame']) if f.endswith('.jpg')])
    frame_path = os.path.join(io_args['video_frame'], frame_files[frame_num])
    ## prepare to add mask
    frame_img = cv2.imread(frame_path)
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    print(frame_num)
    predictor.reset_state(inference_state)
    for obj_id, obj_mask in video_segments[frame_num].items(): # sam2 doesnt use pythonic indexing for UI
        
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state,
                    frame_num,
                    obj_id,
                    obj_mask[0],
        )
    print(video_segments[frame_num].items())
    obj_ids_dict = dict()
    return chosen_frame_img, frame_img, frame_num, obj_ids_dict

def zip_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_STORED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

def choose_obj_to_refine(io_args, frame_num, evt:gr.SelectData):
    # # curr_mask=Seg_Tracker.first_frame_mask
    # frame_path = os.path.join(io_args['output_mask_dir'], f'{frame_num:07d}.png')
    # # chosen_frame_img = Image.open(frame_path).convert('P')
    # chosen_frame_img = cv2.imread(frame_path)
    # chosen_frame_img = cv2.cvtColor(chosen_frame_img, cv2.COLOR_RGB2BGR)
    # print(frame_path[frame_num])
    # idx = chosen_frame_img[evt.index[1],evt.index[0]]
    # curr_idx_mask = np.where(chosen_frame_img == idx, 1, 0).astype(np.uint8)
    
    np_frame_path =os.path.join(io_args['output_mask_dir_np'], f'{frame_num:07d}.npy')
    chosen_frame = np.load(np_frame_path)
    obj_id = chosen_frame[evt.index[1],evt.index[0]]
    chosen_id_mask = np.where(chosen_frame == obj_id, 1, 0).astype(np.uint8)
    
    print(type(obj_id))
    ## get directory of images
    combined_output_path = io_args['output_masked_frame_dir']
    chosen_frame_path = sorted(glob(combined_output_path +'/*.png'))[frame_num]
    total_frames = len(chosen_frame_path)
    chosen_frame_show = cv2.imread(chosen_frame_path)
    chosen_frame_show = cv2.cvtColor(chosen_frame_show, cv2.COLOR_BGR2RGB)
    chosen_frame_show = draw_outline(mask=chosen_id_mask, frame=chosen_frame_show)
    print(f'Object ID: {obj_id}')
        

    return chosen_frame_show, f'Chosen Frame: {frame_num+1}/{total_frames}, Chosen Mask ID:{obj_id}', int(obj_id)

def sam2_app():
    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    app = gr.Blocks()
    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">SAM 2 for Cell Tracking</span>
            </div>
            This is a research demo to track cells using SAM2 as a base. <strong>It is not meant to be used in a commercial setting!</strong>.
            '''
        )
        with gr.Accordion("Instructions", open=False):
            gr.Markdown(
                '''
                ## How to Use?
                1. Upload video file.
                2. Click the "Activate SAM" button to initiate preprocessing.
                3. Click on "Auto Segmentation Mode" to segment majority of object. **Note:** Will miss some object and might segment unintended objects.
                4. Click on "Add new object" and click on the object to track another object.
                5. Refine the initial segmentation by specifying its object ID.
                6. Click on "Start Tracking" to begin tracking.
                7. Download the files and video.  
                
                ## What is the "Choose this frame to refine" button?
                Some objects might not be visible in the initial frame, this function would allow the user to label the new object individually and the new object will be tracked throughout the video.
                1. Choose the frame using the slider; **Note:** Pay attention to "Chosen Frame" textbox as the slider uses percentages!
                2. Click on "Add new object" and click on the new object to track it.   
                3. Click on "Start Tracking" to begin tracking.
                4. Download the files and video.       
                '''
            )
        
        
        # click_stack = gr.State([[],[]]) # Storage clicks status
        origin_frame = gr.State(None)
        predictor = gr.State(None)
        # new_obj_flag = gr.State(None)
        io_args = gr.State(None)
        inference_state = gr.State(None)
        # curr_obj_id= gr.State(None)
        obj_ids_dict = gr.State(dict())
        frame_num = gr.State(value=int(0))
        old_video_segments = gr.State(dict())
        obj_counter = gr.State(value=int(1))
        obj_stack = gr.State(list())
        
        with gr.Row():
            with gr.Column(scale=1):
                # tab_vid_in = gr.Tab(label="Input Video")
                # with tab_vid_in: # add into the tab
                video_input = gr.Video(label='Input video', sources=['upload'])
                
                # seperate video upload and sam initialisation
                preprocess_button = gr.Button(
                                value="Activate SAM!",
                                interactive=True,
                            )
                
                
                first_input_init = gr.Image(label='Segment result of first frame',interactive=True)
                
                
                # tab_click = gr.Tab(label="Manual Click")
                # with tab_click:
                with gr.Row():
                    point_mode = gr.Radio(
                                choices=["Positive",  "Negative"],
                                value="Positive",
                                label="Point Prompt",
                                interactive=True)
                    new_object_button = gr.Button(
                        value="Add new object", 
                        interactive=True
                    )
                    curr_obj_id = gr.Number(
                        value=1, 
                        label='Current Object ID', 
                        interactive=True
                        )
                # tab_auto = gr.Tab(label="Auto Mode")       
                # with tab_auto:
                segm_every = gr.Button(
                    value="Auto Segmentation Mode",
                    interactive=True
                )
                track_for_video = gr.Button(
                            value="Start Tracking",
                                interactive=True,
                                )        
                        
                        
            with gr.Column(scale=1):
                with gr.Accordion("Processed Frames", open=True):
                    processed_frames = gr.Image(label="processed frame", interactive=True)
                    frame_per = gr.Slider(
                        label = "Percentage of Frames Viewed",
                        minimum= 0.0,
                        maximum= 100.0,
                        step=0.01,
                        value=0.0,
                    )
                    obj_select_text = gr.Textbox(value="Please choose a frame to check (in %)!", label='Chosen Frame', interactive=False)
                    frame_per.release(show_res_by_slider, inputs=[frame_per, io_args], outputs=[processed_frames, obj_select_text, frame_num])
                    roll_back_button = gr.Button(value="Choose this frame to refine")
                # video_output = gr.Video(label='Output video', show_download_button= True)
                output_mp4 = gr.File(label="Predicted video")
                output_mask = gr.File(label="Predicted masks")
    ##########################################################
    ######################  back-end #########################
    ##########################################################
    
    # listen to the input_video to get the first frame of video
        preprocess_button.click(
            fn=get_meta_from_video,
            inputs=[
                video_input, 
                predictor, 
                inference_state, 
                old_video_segments, 
                io_args,
                obj_counter
            ],
            outputs=[
                first_input_init, 
                origin_frame, 
                io_args, 
                predictor,
                inference_state,
                obj_counter,
                frame_num,
                obj_ids_dict,
                old_video_segments
            ]
        )
        
        # tab_click.select(
        #     fn=init_sam2,
        #     inputs=[
        #         io_args
        #     ],
        #     outputs=[
        #         predictor,
        #         inference_state, 
        #         curr_obj_id
        #     ],
        #     queue=False,
        # )
        
        new_object_button.click(
            fn=add_new_obj,
            inputs=[
                curr_obj_id,
                obj_counter
                ],
            outputs=[
                curr_obj_id,
                obj_counter
            ]
        )
        first_input_init.select(
           fn= sam_click,
            inputs=[
                predictor,
                origin_frame, 
                inference_state, 
                point_mode, 
                obj_ids_dict, 
                curr_obj_id,
                frame_num,
                obj_stack
            ],
            outputs=[
                predictor, 
                first_input_init, 
                obj_ids_dict,
                obj_stack
            ]
        )
        # segment every object in image
        segm_every.click(
            fn=segment_everything,
            inputs=[
                predictor,
                inference_state,
                origin_frame
            ],
            outputs=[
                predictor,
                first_input_init,
                obj_counter
            ]
        )
        
        # Track object in video
        track_for_video.click(
            fn=begin_track,
            inputs=[
                predictor,
                inference_state,
                io_args, 
                video_input,
                old_video_segments,
                frame_num

            ],
            outputs=[
                processed_frames, output_mp4, output_mask, old_video_segments
            ]
        )
        roll_back_button.click(
            fn=send_to_board,
            inputs=[
                io_args,
                frame_num, 
                old_video_segments, 
                predictor, 
                inference_state,
                obj_ids_dict
            ],
            outputs=[
                first_input_init,
                origin_frame,
                frame_num,
                obj_ids_dict
            ]
        )
        
        processed_frames.select(
            fn=choose_obj_to_refine,
            inputs=[
                io_args, 
                frame_num
            ],
            outputs =[
                processed_frames,
                obj_select_text,
                curr_obj_id
            ]
        )
        
        with gr.Tab(label='Video example'):
            gr.Examples(
                examples=[
                    os.path.join(os.path.dirname(__file__), "assets", "A1-8_5s.mp4"),
                    os.path.join(os.path.dirname(__file__), "assets", "A1_8.mp4")
                    ],
                inputs=[video_input],
            )
    app.queue(default_concurrency_limit=1)
    app.launch(debug=True, share=True)
    
if __name__ == "__main__":
    sam2_app()