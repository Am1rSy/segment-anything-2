from PIL.ImageOps import colorize, scale
import gradio as gr
import cv2
from PIL import Image
import torch
import time, math
import numpy as np
import os,io
from sam2.build_sam import build_sam2_video_predictor
# from skimage.morphology.binary import binary_dilation
import matplotlib.pyplot as plt
import ffmpeg
    
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
        'video_frame': f'{tracking_result_dir}/frame',
        'output_mask_dir': f'{tracking_result_dir}/{video_name}_masks',
        'output_masked_frame_dir': f'{tracking_result_dir}/{video_name}_masked_frames',
        'output_mask_dir_png': f'{tracking_result_dir}/{video_name}_masks/png',
        'output_masked_frame_dir_png': f'{tracking_result_dir}/{video_name}_masked_frames/png',
        'output_mask_dir_np': f'{tracking_result_dir}/{video_name}_masks/numpy',
        'output_masked_frame_dir_np': f'{tracking_result_dir}/{video_name}_masked_frames/numpy',
        # 'output_video': f'{tracking_result_dir}/{video_name}_seg.mp4', # keep same format as input video
        # 'zip_mask': f'{tracking_result_dir}/{video_name}_pred_mask.zip'
    }
    
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
    predictor, inference_state, curr_obj_id = init_sam2(io_args)
    prgrs_bar(1, desc="SAM2 initialised!")
    return first_frame_rgb, first_frame_rgb, io_args, predictor, inference_state, curr_obj_id,""


def init_sam2(io_args):
    sam2_checkpoint = "./checkpoint/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=io_args['video_frame'])
    predictor.reset_state(inference_state)
    
    curr_obj_id = 1
    print("SAM initialised!")
    return predictor, inference_state, curr_obj_id

def get_click_prompt(click_stack, point):

    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"]
    )
    
    prompt = {
        "points_coord":click_stack[0],
        "points_mode":click_stack[1],
        "multimask":"True",
    }

    return prompt

def add_new_obj(obj_ids):
    obj_ids += 1
    return obj_ids
    
    
def show_mask(mask, image=None, obj_id=None):
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    
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
        for c in range(3): 
            image[..., c] = np.where(alpha_mask > 0, (1 - alpha_mask) * image[..., c] + alpha_mask * colored_mask[..., c], image[..., c]) # assign value on pixel location that contain the mask
        return image
    return mask_image

    
def sam_click(predictor, origin_frame, inference_state, point_mode, click_stack, curr_obj_id,evt:gr.SelectData):
    """
    Args:
        origin_frame: nd.array
        click_stack: [[coordinate], [point_mode]]
    """

    print("Click")
    # for mode, `1` means positive click and `0` means negative click
    if point_mode == "Positive":
        point = {"coord":  np.array([[evt.index[0], evt.index[1]]], dtype=np.float32), "mode": np.array([1], np.int32)}
    else:
        # TODO：add everything positive points
        point = {"coord": np.array([[evt.index[0], evt.index[1]]], dtype=np.float32), "mode": np.array([1], np.int32)}

    # get click prompts for sam to predict mask
    # click_prompt = get_click_prompt(click_stack, point)
    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"]
    )
    ann_frame_idx = 0 #assume 0 for now
    
    print(point)
    # add mask according to prompt
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=curr_obj_id,
    points=point["coord"],
    labels=point['mode'],
    )
    masked_frame = origin_frame.copy()
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
    return predictor, masked_frame, click_stack

def begin_track(predictor, inference_state, io_args, input_video):
    
    fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    video_name = os.path.basename(input_video).split('.')[0]
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(height,width)
    cap.release()
    out_path = f'{io_args["tracking_result_dir"]}/{video_name}output.mp4'
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    frame_files = sorted([f for f in os.listdir(io_args['video_frame']) if f.endswith('.jpg')])
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        ## get original frames
        frame_idx = int(os.path.splitext(frame_files[out_frame_idx])[0])
        frame_path = os.path.join(io_args['video_frame'], frame_files[out_frame_idx])
        ## prepare to add mask
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_frame = image.copy()
        
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        ## add mask depending on num of obj in frame
        for obj_id, mask in video_segments[out_frame_idx].items():
                masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
                mask_output_path = os.path.join(io_args['output_mask_dir'], f'{obj_id}_{out_frame_idx:07d}.png')
                cv2.imwrite(mask_output_path, show_mask(mask))
        ## save annotated (masked) image
        combined_output_path = os.path.join(io_args['output_masked_frame_dir'], f'{out_frame_idx:07d}.png')
        combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        yield combined_image_bgr, None, None
        out.write(combined_image_bgr)
        cv2.imwrite(combined_output_path, combined_image_bgr)
        
    out.release()
        
   
    
    zip_path = f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"
    os.system(f"zip -r {zip_path} {io_args['output_mask_dir']}")
    print("done")
    
    return  combined_image_bgr, out_path, zip_path

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
        
        click_stack = gr.State([[],[]]) # Storage clicks status
        origin_frame = gr.State(None)
        predictor = gr.State(None)
        new_obj_flag = gr.State(None)
        io_args = gr.State(None)
        inference_state = gr.State(None)
        curr_obj_id= gr.State(None)
        obj_ids = gr.State(value=(int(0)))
        frame_num = gr.State(value=(int(0)))
        
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
                
                
                tab_click = gr.Tab(label="Click")
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
                        track_for_video = gr.Button(
                            value="Start Tracking",
                                interactive=True,
                                )
                        
            with gr.Column(scale=0.5):
                processed_frames = gr.Image(label="processed frame")
                video_output = gr.Video(label='Output video', show_download_button= True)
                # output_mp4 = gr.File(label="Predicted video")
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
                obj_ids
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
            inputs=[obj_ids],
            outputs=[
                obj_ids
            ]
        )
        first_input_init.select(
           fn= sam_click,
            inputs=[
                predictor,
                origin_frame, 
                inference_state, 
                point_mode, 
                click_stack, 
                obj_ids
            ],
            outputs=[
                predictor, 
                first_input_init, 
                click_stack
            ]
        )
        
        # Track object in video
        track_for_video.click(
            fn=begin_track,
            inputs=[
                predictor,
                inference_state,
                io_args, 
                video_input
            ],
            outputs=[
                processed_frames, video_output, output_mask
            ]
        )
        
    app.queue(default_concurrency_limit=1)
    app.launch(debug=True, share=True)
    
if __name__ == "__main__":
    sam2_app()