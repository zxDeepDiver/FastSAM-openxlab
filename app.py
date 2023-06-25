from ultralytics import YOLO
import gradio as gr
import torch
from tools import fast_process

# Load the pre-trained model
model = YOLO('checkpoints/FastSAM.pt')

# Description
title = "<center><strong><font size='8'>🏃 Fast Segment Anything 🤗</font></strong></center>"

news = """ # News

        🔥 Add the 'Advanced options" in Everything mode to get a more detailed adjustment.
        """

         
        # 🔥 Support the points mode and box mode, text mode will come soon.

description = """This is a demo on Github project 🏃 [Fast Segment Anything Model](https://github.com/CASIA-IVA-Lab/FastSAM).
                
                🎯 Upload an Image, segment it with Fast Segment Anything (Everything mode). The other modes will come soon.
                
                ⌛️ It takes about 4~ seconds to generate segment results. The concurrency_count of queue is 1, please wait for a moment when it is crowded.
                
                🚀 To get faster results, you can use a smaller input size and leave high_visual_quality unchecked.
                
                📣 You can also obtain the segmentation results of any Image through this Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing)
                
                😚 A huge thanks goes out to the @HuggingFace Team for supporting us with GPU grant.
                
                🏠 Check out our [Model Card 🏃](https://huggingface.co/An-619/FastSAM)
                
              """

examples = [["assets/sa_8776.jpg"], ["assets/sa_414.jpg"], ["assets/sa_1309.jpg"], ["assets/sa_11025.jpg"],
            ["assets/sa_561.jpg"], ["assets/sa_192.jpg"], ["assets/sa_10039.jpg"], ["assets/sa_862.jpg"]]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def segment_image(
    input,
    input_size=1024, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    mask_random_color=True,
    withContours=True,
    points=None,
    bbox=None,
    point_label=None,
    use_retina=True,
    ):
    input_size = int(input_size)  # 确保 imgsz 是整数

    # Thanks for the suggestion by hysts in HuggingFace.
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)
    fig = fast_process(annotations=results[0].masks.data,
                        image=input,
                        device=device,
                        scale=(1024 // input_size),
                        better_quality=better_quality,
                        mask_random_color=mask_random_color,
                        points=points,
                        bbox=bbox,
                        point_label=point_label,
                        use_retina=use_retina,
                        withContours=withContours,)
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cond_img = gr.Image(label="Input", value=default_example[0], type='pil')

segm_img = gr.Image(label="Segmented Image", interactive=False, type='pil')

input_size_slider = gr.components.Slider(minimum=512,
                                         maximum=1024,
                                         value=1024,
                                         step=64,
                                         label='Input_size (Our model was trained on a size of 1024)')

with gr.Blocks(css=css, title='Fast Segment Anything') as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)
    
        with gr.Column(scale=1):
            # News
            gr.Markdown(news)

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
                contour_check = gr.Checkbox(value=True, label='withContours')

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

        with gr.Column():
            with gr.Accordion("Advanced options", open=False):
                iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou_threshold')
                conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='conf_threshold')
                mor_check = gr.Checkbox(value=False, label='better_visual_quality')
                
            # Description
            gr.Markdown(description)

    segment_btn.click(segment_image,
                    inputs=[cond_img, input_size_slider, iou_threshold, conf_threshold, mor_check, contour_check],
                    outputs=segm_img)

    # def clear():
    # return None, None

    # clear_btn.click(fn=clear, inputs=None, outputs=None)

demo.queue()
demo.launch()
