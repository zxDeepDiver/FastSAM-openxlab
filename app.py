from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import cv2
import torch

model = YOLO('checkpoints/FastSAM.pt')  # load a custom model


def fast_process(annotations, image):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    #original_h = image.shape[0]
    #original_w = image.shape[1]
    #for i, mask in enumerate(annotations):
    #        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    #        annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
    fast_show_mask(annotations,
                       plt.gca())
                       #target_height=original_h,
                       #target_width=original_w)
    plt.axis('off')
    plt.tight_layout()
    return fig


#   CPU post process
def fast_show_mask(annotation, ax):
    msak_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    # 将annotation 按照面积 排序
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::1]
    annotation = annotation[sorted_indices]

    index = (annotation != 0).argmax(axis=0)
    color = np.random.random((msak_sum, 1, 1, 3))
    transparency = np.ones((msak_sum, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    show = np.zeros((height, weight, 4))

    h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    # 使用向量化索引更新show的值
    show[h_indices, w_indices, :] = mask_image[indices]


    #if retinamask == False:
    #    show = cv2.resize(show, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    ax.imshow(show)



# post_process(results[0].masks, Image.open("../data/cake.png"))

def predict(input, input_size=512):
    input_size = int(input_size)  # 确保 imgsz 是整数
    results = model(input, device='cpu', retina_masks=True, iou=0.7, conf=0.25, imgsz=input_size)
    pil_image = fast_process(annotations=results[0].masks.data.numpy(), image=input)

    return pil_image


# inp = 'assets/sa_192.jpg'
# results = model(inp, device='cpu', retina_masks=True, iou=0.7, conf=0.25, imgsz=1024)
# results = format_results(results[0], 100)
# post_process(annotations=results, image_path=inp)

demo = gr.Interface(fn=predict,
                    inputs=[gr.inputs.Image(type='pil'), gr.inputs.Dropdown(choices=[512, 800, 1024], default=512)],
                    outputs=['plot'],
                    examples=[["assets/sa_8776.jpg", 1024]],
                    #    ["assets/sa_1309.jpg", 1024]],
                    # examples=[["assets/sa_192.jpg"], ["assets/sa_414.jpg"],
                    #           ["assets/sa_561.jpg"], ["assets/sa_862.jpg"],
                    #           ["assets/sa_1309.jpg"], ["assets/sa_8776.jpg"],
                    #           ["assets/sa_10039.jpg"], ["assets/sa_11025.jpg"],],
                    )

demo.launch()
"""

from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import torch

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
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, mask in enumerate(annotations):
        show_mask(mask, plt.gca(),random_color=mask_random_color,bbox=bbox,points=points)
    plt.axis('off')
    
    plt.tight_layout()
    return fig


# post_process(results[0].masks, Image.open("../data/cake.png"))

def predict(input, input_size):
    input_size = int(input_size)  # 确保 imgsz 是整数
    results = model(input, device='cpu', retina_masks=True, iou=0.7, conf=0.25, imgsz=input_size)
    results = format_results(results[0], 100)
    results.sort(key=lambda x: x['area'], reverse=True)
    pil_image = post_process(annotations=results, image=input)
    return pil_image

# inp = 'assets/sa_192.jpg'
# results = model(inp, device='cpu', retina_masks=True, iou=0.7, conf=0.25, imgsz=1024)
# results = format_results(results[0], 100)
# post_process(annotations=results, image_path=inp)

demo = gr.Interface(fn=predict,
                    inputs=[gr.inputs.Image(type='pil'), gr.inputs.Dropdown(choices=[512, 800, 1024], default=1024)],
                    outputs=['plot'],
                     examples=[["assets/sa_8776.jpg", 1024]],
                            #    ["assets/sa_1309.jpg", 1024]],
                    # examples=[["assets/sa_192.jpg"], ["assets/sa_414.jpg"],
                    #           ["assets/sa_561.jpg"], ["assets/sa_862.jpg"],
                    #           ["assets/sa_1309.jpg"], ["assets/sa_8776.jpg"],
                    #           ["assets/sa_10039.jpg"], ["assets/sa_11025.jpg"],],
                    )

demo.launch()

"""