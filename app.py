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