import argparse
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
import torch
from qwen_vl_utils import process_vision_info
import yaml
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json


def process_image(image_path):
    # 处理图像，使其可以输入模型
    image = Image.open(image_path).convert("RGB")
    return image


def bot_inference(prompt, image_path, generation_args):
    
    # 生成模型的输入
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user", 

                "content": [
                    {"type": "image", "image": 'data/coco_3d/train2017/000000144884.jpg'},
                    {"type": "depth", "image": 'data/coco_3d/depth/000000144884_remove_edges.png'},
                    {"type": "camera_parameters", "image": 'data/coco_3d/camera_parameters/000000144884.json'},
                    {"type": "text", "text": prompt},
                ]
            }
        ], 
        tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(
        [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": 'data/coco_3d/train2017/000000144884.jpg'},
                    {"type": "depth", "image": 'data/coco_3d/depth/000000144884_remove_edges.png'},
                    {"type": "camera_parameters", "image": 'data/coco_3d/camera_parameters/000000144884.json'},
                ]
            }
        ]
    )

    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    
    # 进行推理
    streamer = TextIteratorStreamer(processor.tokenizer, **{"skip_special_tokens": True, "skip_prompt": True, 'clean_up_tokenization_spaces': False})
    generation_kwargs = dict(inputs, streamer=streamer, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    output = model.generate(**generation_kwargs)
    
    # 输出结果
    result = processor.tokenizer.decode(output[0], skip_special_tokens=True)

    print(result)

    
    
def main(args):
    global processor, model, device

    device = args.device
    
    disable_torch_init()

    use_flash_attn = True
    
    model_name = get_model_name_from_path(args.model_path)
    
    if args.disable_flash_attention:
        use_flash_attn = False

    # 加载模型
    processor, model = load_pretrained_model(
        model_base=args.model_base, model_path=args.model_path, 
        device_map=args.device, model_name=model_name, 
        load_4bit=args.load_4bit, load_8bit=args.load_8bit,
        device=args.device, use_flash_attn=use_flash_attn
    )

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }

    prompt = """What can be seen regarding the vase in the 3D scene?\n<spatial_image>"""
    image_path = "./inference_demo/math4.jpeg"
    
    bot_inference(prompt, image_path, generation_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
