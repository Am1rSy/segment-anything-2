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
from glob import glob
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# from skimage.morphology.binary import binary_dilation
import matplotlib.pyplot as plt
import ffmpeg
import shutil

def delete_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def get_meta_from_video(input_video, prgrs_bar=gr.Progress()):
    if input_video is None:
        return None, None,""
    prgrs_bar(0, desc="Starting")
#     if predictor is not None:
#         predictor.reset_state(inference_state)
    
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
    vidcap.release()
    # for frame_idx in prgrs_bar.tqdm(range(total_frames), desc="Extracting frames..."):
    frame_interval = 1
    print(frame_interval)
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
    prgrs_bar(0.7, desc="Uploading frames to SAM2!")
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    predictor, inference_state = init_sam2(io_args)
    prgrs_bar(1, desc="SAM2 initialised!")
    return first_frame_rgb, first_frame_rgb, io_args, predictor, inference_state, ""


def init_sam2(io_args):
    # sam2_checkpoint = "./checkpoint/sam2_hiera_large.pt"
    # model_cfg = "sam2_hiera_l.yaml"
    sam2_checkpoint = "./checkpoint/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=io_args['video_frame'], offload_state_to_cpu=True, async_loading_frames=True)
    predictor.reset_state(inference_state)
    
    # curr_obj_id = dict()
    print("SAM initialised!")
    return predictor, inference_state

def init_sam2_everything():
    # sam2_checkpoint = "./checkpoint/sam2_hiera_large.pt"
    # model_cfg = "sam2_hiera_l.yaml"
    sam2_checkpoint = "./checkpoint/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"
    
    sam2_everything = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_everything)
    return mask_generator

# def get_click_prompt(click_stack, point):

#     click_stack[0].append(point["coord"])
#     click_stack[1].append(point["mode"]
#     )
    
#     prompt = {
#         "points_coord":click_stack[0],
#         "points_mode":click_stack[1],
#         "multimask":"True",
#     }

#     return prompt

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
    center =int(np.mean(pos_y)),int(np.mean(pos_x))
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
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = (mask_image * 255).astype(np.uint8)  # scale colour with 255
    
    ## more elegant way to overlay image and mask (ref: medSAM)
    if image is not None:
        image_h, image_w = image.shape[:2]
        if (image_h, image_w) != (h, w):
            raise ValueError(f"Image dimensions ({image_h}, {image_w}) and mask dimensions ({h}, {w}) do not match")
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3): # iterate through the mask and add into combined image
            colored_mask[..., c] = mask_image[..., c]
        alpha_mask = mask_image[..., 3] / 255.0 #scale mask pixel value to normal
        # print(np.unique(mask))
        for c in range(3): 
            image[..., c] = np.where(alpha_mask > 0, (1 - alpha_mask) * image[..., c] + alpha_mask * colored_mask[..., c], image[..., c]) # assign value on pixel location that contain the mask
       
        return image
    return mask_image

    
def sam_click(predictor, origin_frame, inference_state, point_mode, obj_ids_dict, curr_obj_id,frame_num, obj_stack,evt:gr.SelectData):
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
    return predictor, masked_frame, obj_ids_dict, obj_stack

def segment_everything(predictor, inference_state, origin_frame):
    mask_generator = init_sam2_everything()
    masks = mask_generator.generate(origin_frame)
    
    ## remove background annotations
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    sorted_anns.pop(0)
    # img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    # img[:,:,3] = 0
    # lowres_side_length = predictor.image_size // 4
    dtype = next(predictor.parameters()).dtype
    device = "cuda"
    for mask_idx, mask_results in enumerate(sorted_anns):
        # Get mask into form expected by the model
        mask_tensor = torch.tensor(mask_results["segmentation"], dtype=dtype, device=device)
        # lowres_mask = torch.nn.functional.interpolate(
        #     mask_tensor.unsqueeze(0).unsqueeze(0),
        #     size=(lowres_side_length, lowres_side_length),
        #     mode="bilinear",
        #     align_corners=False,
        # ).squeeze()

        # Add each mask as it's own 'object' to segment
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=mask_idx+1, # add 1 to prevent start from 0
            mask=mask_tensor,
        )
        
    masked_frame = origin_frame.copy()
    # print(obj_ids_dict[curr_obj_id])
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
        masked_frame = labelled_mask(mask,masked_frame, obj_id)
    print(len(out_obj_ids))
    del mask_generator
    return predictor, masked_frame, len(out_obj_ids)
        
    
    

def begin_track(predictor, inference_state, io_args, input_video, old_video_segments, frame_num):
    
    fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    video_name = os.path.basename(input_video).split('.')[0]
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    out_path = f'{io_args["tracking_result_dir"]}/{video_name}_output.mp4'
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    

    ## add non-modified frame into video
    if frame_num != 0: # check if adding new masks
        start_frame_idx = frame_num
        print(start_frame_idx)  
        for out_frame_idx in range(0, start_frame_idx):
            combined_output_path = os.path.join(io_args['output_masked_frame_dir'], f'{out_frame_idx:07d}.png')
        ## prepare to add mask
            combined_image_bgr = cv2.imread(combined_output_path)
            out.write(combined_image_bgr)
    
    temp_video_segments = copy.deepcopy(old_video_segments) # need to make copy to prevent same reference with new
    frame_files = sorted([f for f in os.listdir(io_args['video_frame']) if f.endswith('.jpg')])
    # run propagation throughout the video and collect the results in a dict
    if old_video_segments:
        new_video_segments = old_video_segments  # video_segments contains the per-frame segmentation results (frame id: {obj_id: (1,H,W)})
    else:
        new_video_segments ={}
        
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        ## get original frames
        # frame_idx = int(os.path.splitext(frame_files[out_frame_idx])[0])
        frame_path = os.path.join(io_args['video_frame'], frame_files[out_frame_idx])
        ## prepare to add mask
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_frame = image.copy()
        new_video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
        ## combine future mask addons
        if temp_video_segments is not None and temp_video_segments.get(out_frame_idx):
            new_video_segments[out_frame_idx] = temp_video_segments[out_frame_idx] | new_video_segments[out_frame_idx]
        
        mask_output = np.zeros_like(masked_frame)
        mask_np_output = np.zeros_like(masked_frame[...,0])
        # print(mask_np_output.shape)
        ## add mask depending on num of obj in frame
        for obj_id, mask in new_video_segments[out_frame_idx].items():
                masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
                mask_output = show_mask(mask, image=mask_output, obj_id=obj_id)
                mask_np_output[mask[0]==True] = obj_id
                masked_frame = labelled_mask(mask,masked_frame, obj_id)
        yield masked_frame, None, None, None
        ## save annotated (masked) image
        mask_output_path = os.path.join(io_args['output_mask_dir_png'], f'{out_frame_idx:07d}.png')
        mask_output = cv2.cvtColor(mask_output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_output_path, mask_output)
        
        
        combined_output_path = os.path.join(io_args['output_masked_frame_dir'], f'{out_frame_idx:07d}.png')
        combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        # combined_image_bgr = masked_frame
        
        
        out.write(combined_image_bgr)
        cv2.imwrite(combined_output_path, combined_image_bgr)
        np_file = np.array(mask_np_output)
        np.save(f"{io_args['output_mask_dir_np']}/{out_frame_idx:07d}.npy", np_file)
        
    out.release()
    #zip files
    zip_path = f"{io_args['tracking_result_dir']}/{video_name}.zip"
    zip_folder(f"{io_args['tracking_result_dir']}/outputs", zip_path)

    # os.system(f"zip -r {zip_path} {io_args['output_mask_dir']}")
    print("done")
    
    yield cv2.cvtColor(combined_image_bgr, cv2.COLOR_BGR2RGB), out_path, zip_path, new_video_segments

def show_res_by_slider(frame_percent, io_args):
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
                <span style="font-size:3em; font-weight:bold;">SAM 2 Demo</span>
            </div>
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
            with gr.Column(scale=0.5):
                # tab_vid_in = gr.Tab(label="Input Video")
                # with tab_vid_in: # add into the tab
                video_input = gr.Video(label='Input video', sources=['upload'])
                
                # seperate video upload and sam initialisation
                preprocess_button = gr.Button(
                                value="Activate SAM!",
                                interactive=True,
                            )
                
                
                first_input_init = gr.Image(label='Segment result of first frame',interactive=True)
                
                
                tab_click = gr.Tab(label="Manual Click")
                with tab_click:
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
                tab_auto = gr.Tab(label="Auto Mode")       
                with tab_auto:
                    segm_every = gr.Button(
                        value="Auto Segmentation Mode",
                        interactive=True
                    )
                track_for_video = gr.Button(
                            value="Start Tracking",
                                interactive=True,
                                )        
                        
                        
            with gr.Column(scale=0.5):
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
                video_input
            ],
            outputs=[
                first_input_init, 
                origin_frame, 
                io_args, 
                predictor,
                inference_state, 
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
                    ],
                inputs=[video_input],
            )
    app.queue(default_concurrency_limit=1)
    app.launch(debug=True, share=True)
    
if __name__ == "__main__":
    sam2_app()