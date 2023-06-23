from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import cv2
import torch
from PIL import Image

# Load the pre-trained model
model = YOLO('checkpoints/FastSAM.pt')

# Description
title = "<center><strong><font size='8'>🏃 Fast Segment Anything 🤗</font></strong></center>"

description = """This is a demo on Github project 🏃 [Fast Segment Anything Model](https://github.com/CASIA-IVA-Lab/FastSAM).
                
                🎯 Upload an Image, segment it with Fast Segment Anything (Everything mode). The other modes will come soon.
                
                ⌛️ It takes about 4~ seconds to generate segment results. The concurrency_count of queue is 1, please wait for a moment when it is crowded.
                
                🚀 To get faster results, you can use a smaller input size and leave high_visual_quality unchecked.
                
                📣 You can also obtain the segmentation results of any Image through this Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing)
                
                😚 A huge thanks goes out to the @HuggingFace Team for supporting us with GPU grant.
                
                🏠 Check out our [Model Card 🏃](https://huggingface.co/An-619/FastSAM)
                
              """

examples = [["assets/sa_8776.jpg"], ["assets/sa_414.jpg"],
            ["assets/sa_1309.jpg"], ["assets/sa_11025.jpg"],
            ["assets/sa_561.jpg"], ["assets/sa_192.jpg"],
            ["assets/sa_10039.jpg"], ["assets/sa_862.jpg"]]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

def fast_process(annotations, image, high_quality, device, scale):
    if isinstance(annotations[0],dict):
        annotations = [annotation['segmentation'] for annotation in annotations]

    original_h = image.height
    original_w = image.width
    if high_quality == True:
        if isinstance(annotations[0],torch.Tensor):
            annotations = np.array(annotations.cpu())
        for i, mask in enumerate(annotations):
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
    if device == 'cpu':
        annotations = np.array(annotations)
        inner_mask = fast_show_mask(annotations,
                       plt.gca(),
                       bbox=None,
                       points=None,
                       pointlabel=None,
                       retinamask=True,
                       target_height=original_h,
                       target_width=original_w)
    else:
        if isinstance(annotations[0],np.ndarray):
            annotations = torch.from_numpy(annotations)
        inner_mask = fast_show_mask_gpu(annotations,
                           plt.gca(),
                           bbox=None,
                           points=None,
                           pointlabel=None)
    if isinstance(annotations, torch.Tensor):
        annotations = annotations.cpu().numpy()
    
    if high_quality:
        contour_all = []
        temp = np.zeros((original_h, original_w,1))
        for i, mask in enumerate(annotations):
            if type(mask) == dict:
                mask = mask['segmentation']
            annotation = mask.astype(np.uint8)
            contours, _ = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_all.append(contour)
        cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2 // scale)
        color = np.array([0 / 255, 0 / 255, 255 / 255, 0.9])
        contour_mask = temp / 255 * color.reshape(1, 1, -1)
    image = image.convert('RGBA')
    
    overlay_inner = Image.fromarray((inner_mask * 255).astype(np.uint8), 'RGBA')
    image.paste(overlay_inner, (0, 0), overlay_inner)
    
    if high_quality:
        overlay_contour = Image.fromarray((contour_mask * 255).astype(np.uint8), 'RGBA')
        image.paste(overlay_contour, (0, 0), overlay_contour)
        
    return image

#   CPU post process
def fast_show_mask(annotation, ax, bbox=None, 
                   points=None, pointlabel=None,
                   retinamask=True, target_height=960,
                   target_width=960):
    msak_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    # 将annotation 按照面积 排序
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::1]
    annotation = annotation[sorted_indices]

    index = (annotation != 0).argmax(axis=0)
    color = np.random.random((msak_sum,1,1,3))
    transparency = np.ones((msak_sum,1,1,1)) * 0.6
    visual = np.concatenate([color,transparency],axis=-1)
    mask_image = np.expand_dims(annotation,-1) * visual

    mask = np.zeros((height,weight,4))

    h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    # 使用向量化索引更新show的值
    mask[h_indices, w_indices, :] = mask_image[indices]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
    # draw point
    if points is not None:
        plt.scatter([point[0] for i, point in enumerate(points) if pointlabel[i]==1], [point[1] for i, point in enumerate(points) if pointlabel[i]==1], s=20, c='y')
        plt.scatter([point[0] for i, point in enumerate(points) if pointlabel[i]==0], [point[1] for i, point in enumerate(points) if pointlabel[i]==0], s=20, c='m')
    
    if retinamask==False:
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    
    return mask


def fast_show_mask_gpu(annotation, ax,
                       bbox=None, points=None, 
                       pointlabel=None):
    msak_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    # 找每个位置第一个非零值下标
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    color = torch.rand((msak_sum,1,1,3)).to(annotation.device)
    transparency = torch.ones((msak_sum,1,1,1)).to(annotation.device) * 0.6
    visual = torch.cat([color,transparency],dim=-1)
    mask_image = torch.unsqueeze(annotation,-1) * visual
    # 按index取数，index指每个位置选哪个batch的数，把mask_image转成一个batch的形式
    mask = torch.zeros((height,weight,4)).to(annotation.device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    # 使用向量化索引更新show的值
    mask[h_indices, w_indices, :] = mask_image[indices]
    mask_cpu = mask.cpu().numpy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
    # draw point
    if points is not None:
        plt.scatter([point[0] for i, point in enumerate(points) if pointlabel[i]==1], [point[1] for i, point in enumerate(points) if pointlabel[i]==1], s=20, c='y')
        plt.scatter([point[0] for i, point in enumerate(points) if pointlabel[i]==0], [point[1] for i, point in enumerate(points) if pointlabel[i]==0], s=20, c='m')
    return mask_cpu


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def segment_image(input, input_size=1024, high_visual_quality=True, iou_threshold=0.7, conf_threshold=0.25):
    input_size = int(input_size)  # 确保 imgsz 是整数
    
    # Thanks for the suggestion by hysts in HuggingFace.
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))
    
    results = model(input, device=device, retina_masks=True, iou=iou_threshold, conf=conf_threshold, imgsz=input_size)
    fig = fast_process(annotations=results[0].masks.data,
                        image=input, high_quality=high_visual_quality,
                        device=device, scale=(1024 // input_size))
    return fig

# input_size=1024
# high_quality_visual=True
# inp = 'assets/sa_192.jpg'
# input = Image.open(inp)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# input_size = int(input_size)  # 确保 imgsz 是整数
# results = model(input, device=device, retina_masks=True, iou=0.7, conf=0.25, imgsz=input_size)
# pil_image = fast_process(annotations=results[0].masks.data,
#                             image=input, high_quality=high_quality_visual, device=device)

cond_img = gr.Image(label="Input", value=default_example[0], type='pil')

segm_img = gr.Image(label="Segmented Image", interactive=False, type='pil')

input_size_slider = gr.components.Slider(minimum=512, maximum=1024, value=1024, step=64, label='Input_size (Our model was trained on a size of 1024)')

with gr.Blocks(css=css, title='Fast Segment Anything') as demo:
    with gr.Row():
        # Title
        gr.Markdown(title)
    #     # # Description
    #     # gr.Markdown(description)
        
    # Images
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            cond_img.render()
            
        with gr.Column(scale=1):
            segm_img.render()
    
    # Submit & Clear
    with gr.Row():
        with gr.Column():
            input_size_slider.render()
            
            with gr.Row():
                vis_check = gr.Checkbox(value=True, label='high_visual_quality')
                
                with gr.Column():
                    segment_btn = gr.Button("Segment Anything", variant='primary')
                    
                # with gr.Column():
                    # clear_btn = gr.Button("Clear", variant="primary")
            
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(examples=examples,
                        inputs=[cond_img],
                        outputs=segm_img,
                        fn=segment_image,
                        cache_examples=True,
                        examples_per_page=4)
            # gr.Markdown("Try some of the examples below ⬇️")
            # gr.Examples(examples=examples,
            #             inputs=[cond_img, input_size_slider, vis_check, iou_threshold, conf_threshold],
            #             outputs=output,
            #             fn=segment_image,
            #             examples_per_page=4)

        with gr.Column():
            with gr.Accordion("Advanced options", open=False):
                iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou_threshold')
                conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='conf_threshold')
                
            # Description
            gr.Markdown(description)
    
    segment_btn.click(segment_image,
                     inputs=[cond_img, input_size_slider, vis_check, iou_threshold, conf_threshold],
                     outputs=segm_img)  
    
    # def clear():
        # return None, None

    # clear_btn.click(fn=clear, inputs=None, outputs=None)

demo.queue()
demo.launch()

# app_interface = gr.Interface(fn=predict,
#                     inputs=[gr.Image(type='pil'),
#                             gr.components.Slider(minimum=512, maximum=1024, value=1024, step=64, label='input_size'),
#                             gr.components.Checkbox(value=True, label='high_visual_quality')],
#                     # outputs=['plot'],
#                     outputs=gr.Image(type='pil'),
#                     # examples=[["assets/sa_8776.jpg"]],
#                     # #    ["assets/sa_1309.jpg", 1024]],
#                     examples=[["assets/sa_192.jpg"], ["assets/sa_414.jpg"],
#                               ["assets/sa_561.jpg"], ["assets/sa_862.jpg"],
#                               ["assets/sa_1309.jpg"], ["assets/sa_8776.jpg"],
#                               ["assets/sa_10039.jpg"], ["assets/sa_11025.jpg"],],
#                     cache_examples=True,
#                     title="Fast Segment Anything (Everything mode)"
#                     )


# app_interface.queue(concurrency_count=1, max_size=20)
# app_interface.launch()