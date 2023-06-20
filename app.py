from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import io
import torch
# import cv2

model = YOLO('checkpoints/FastSAM.pt')  # load a custom model

def format_results(result,filter = 0):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

    
        if torch.sum(mask) < filter:
            continue
        annotation['id'] = i
        annotation['segmentation'] = mask.cpu().numpy()
        annotation['bbox'] = result.boxes.data[i]
        annotation['score'] = result.boxes.conf[i]
        annotation['area'] = annotation['segmentation'].sum()
        annotations.append(annotation)
    return annotations

def show_mask(annotation, ax, random_color=True, bbox=None, points=None):
    if random_color :    # random mask color
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    if type(annotation) == dict:
        annotation = annotation['segmentation']
    mask = annotation
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # draw box
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
    # draw point
    if points is not None:
        ax.scatter([point[0] for point in points], [point[1] for point in points], s=10, c='g')
    ax.imshow(mask_image)
    return mask_image

def post_process(annotations, image, mask_random_color=True, bbox=None, points=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, mask in enumerate(annotations):
        show_mask(mask, plt.gca(),random_color=mask_random_color,bbox=bbox,points=points)
    plt.axis('off')
    # create a BytesIO object
    buf = io.BytesIO()

    # save plot to buf
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0)
    
    # use PIL to open the image
    img = Image.open(buf)
    
    # copy the image data
    img_copy = img.copy()
    
    # don't forget to close the buffer
    buf.close()
    return img_copy


# def show_mask(annotation, ax, random_color=False):
#     if random_color :    # 掩膜颜色是否随机决定
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
#     mask = annotation.cpu().numpy()
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

# def post_process(annotations, image):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     for i, mask in enumerate(annotations):
#         show_mask(mask.data, plt.gca(),random_color=True)
#     plt.axis('off')
    
    # 获取渲染后的像素数据并转换为PIL图像
    
    return pil_image


# post_process(results[0].masks, Image.open("../data/cake.png"))

def predict(inp):
    results = model(inp, device='cpu', retina_masks=True, iou=0.7, conf=0.25, imgsz=1024)
    results = format_results(results[0], 100)
    pil_image = post_process(annotations=results, image=inp)
    return pil_image

# inp = 'assets/sa_192.jpg'
# results = model(inp, device='cpu', retina_masks=True, iou=0.7, conf=0.25, imgsz=1024)
# results = format_results(results[0], 100)
# post_process(annotations=results, image_path=inp)

demo = gr.Interface(fn=predict,
                    inputs=gr.inputs.Image(type='pil'),
                    outputs=gr.outputs.Image(type='pil'),
                    examples=[["assets/sa_192.jpg"], ["assets/sa_414.jpg"],
                              ["assets/sa_561.jpg"], ["assets/sa_862.jpg"],
                              ["assets/sa_1309.jpg"], ["assets/sa_8776.jpg"],
                              ["assets/sa_10039.jpg"], ["assets/sa_11025.jpg"],],
                    )

demo.launch()