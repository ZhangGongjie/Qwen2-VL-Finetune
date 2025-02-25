import argparse
from threading import Thread
import gradio as gr
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
from functools import partial
import warnings
from qwen_vl_utils import process_vision_info
from src.training.utils_3d import get_coord3d_info, coord3d_to_flat_patches
import torch
from src.training.monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward

warnings.filterwarnings("ignore")

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def process_depth_and_camera(image_file, depth_file, cam_file, min_pixel, max_pixel):
    image_input = get_image_info(image_file, min_pixel, max_pixel)
    coord3d = get_coord3d_info(image_file, depth_file, cam_file, image_input)
    patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size
    temporal_patch_size = processor.image_processor.temporal_patch_size
    coord3d = coord3d_to_flat_patches(coord3d, patch_size, merge_size, temporal_patch_size)
    return coord3d

def get_image_info(image_path, min_pixel, max_pixel):
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixel": min_pixel,
                "max_pixel": max_pixel
            }
         ]
        }
    ]
    image_input, _ = process_vision_info(messages)
    return image_input[0]

def bot_streaming(message, history, generation_args):
    images = []
    videos = []
    depths = []
    camera_params = []

    if "files" in message and message["files"]:
        for file_item in message["files"]:
            file_path = file_item["name"] if isinstance(file_item, dict) and "name" in file_item else file_item
            if is_video_file(file_path):
                videos.append(file_path)
            else:
                images.append(file_path)

    if "depth" in message and message["depth"]:
        for file_item in message["depth"]:
            file_path = file_item["name"] if isinstance(file_item, dict) and "name" in file_item else file_item
            depths.append(file_path)

    if "camera_parameters" in message and message["camera_parameters"]:
        for file_item in message["camera_parameters"]:
            file_path = file_item["name"] if isinstance(file_item, dict) and "name" in file_item else file_item
            camera_params.append(file_path)

    coord3d_list = []
    min_pixel = 256 * 28 * 28
    max_pixel = 1280 * 28 * 28
    if depths and camera_params and images:
        for i, depth_file in enumerate(depths):
            image_file = images[i] if i < len(images) else images[0]
            cam_file = camera_params[i] if i < len(camera_params) else camera_params[0]
            coord3d = process_depth_and_camera(image_file, depth_file, cam_file, min_pixel, max_pixel)
            coord3d = torch.tensor(coord3d)
            coord3d_list.append(coord3d)

    conversation = []
    for user_turn, assistant_turn in history if history else []:
        user_content = []
        if isinstance(user_turn, tuple):
            file_paths = user_turn[0]
            user_text = user_turn[1]
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            for file_path in file_paths:
                if is_video_file(file_path):
                    user_content.append({"type": "video", "video": file_path, "fps": 1.0})
                else:
                    user_content.append({"type": "image", "image": file_path})
            if user_text:
                user_content.append({"type": "text", "text": user_text})
        else:
            user_content.append({"type": "text", "text": user_turn})
        conversation.append({"role": "user", "content": user_content})

        if assistant_turn is not None:
            assistant_content = [{"type": "text", "text": assistant_turn}]
            conversation.append({"role": "assistant", "content": assistant_content})

    user_content = []
    for image in images:
        user_content.append({"type": "image", "image": image})
    for video in videos:
        user_content.append({"type": "video", "video": video, "fps": 1.0})
    if "text" in message and message["text"]:
        user_content.append({"type": "text", "text": message["text"]})
    conversation.append({"role": "user", "content": user_content})

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    if len(depths) > 0:
        inputs["coord3d"] = torch.cat(coord3d_list, dim=0)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
        clean_up_tokenization_spaces=False,
    )
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    if history is None:
        history = []
    if len(history) == 0 or (len(history) > 0 and history[-1][1] != ""):
        user_text = message.get("text", "")
        history.append((user_text, ""))

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        history[-1] = (history[-1][0], buffer)
        yield history


def main(args):
    global processor, model, device

    device = args.device
    
    disable_torch_init()
    use_flash_attn = True
    model_name = get_model_name_from_path(args.model_path)
    
    if args.disable_flash_attention:
        use_flash_attn = False

    replace_qwen2_5_with_mixed_modality_forward()

    processor, model = load_pretrained_model(
        model_base=args.model_base,
        model_path=args.model_path,
        device_map=args.device,
        model_name=model_name,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        device=args.device,
        use_flash_attn=use_flash_attn
    )
    
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }
    
    bot_streaming_with_args = partial(bot_streaming, generation_args=generation_args)
    
    with gr.Blocks() as demo:
        with gr.Row():
            chatbot = gr.Chatbot(scale=2)
        with gr.Row():
            file_input = gr.File(label="Upload image/video files", file_count="multiple", file_types=["image", "video"])
            depth_input = gr.File(label="Upload Depth File", file_count="multiple")
            camera_input = gr.File(label="Upload camera parameter file", file_count="multiple")
        text_input = gr.Textbox(placeholder="Please enter text...", label="Text Input")
        submit_btn = gr.Button("submit")
        
        hidden_message = gr.State()
        
        def package_inputs(file_list, depth_list, camera_list, text):
            return {
                "files": file_list if file_list is not None else [],
                "depth": depth_list if depth_list is not None else [],
                "camera_parameters": camera_list if camera_list is not None else [],
                "text": text if text is not None else ""
            }
        
        submit_btn.click(
            fn=package_inputs,
            inputs=[file_input, depth_input, camera_input, text_input],
            outputs=hidden_message
        ).then(
            bot_streaming_with_args,
            inputs=[hidden_message, chatbot],
            outputs=chatbot
        )

    
    demo.queue(api_open=False)
    demo.launch(show_api=False, share=False, server_name='0.0.0.0')

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
