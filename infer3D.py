import argparse
import warnings
from PIL import Image
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
from peft import PeftModel
import os
from qwen_vl_utils import process_vision_info
from src.training.utils_3d import get_coord3d_info, coord3d_to_flat_patches
from src.training.monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward

warnings.filterwarnings("ignore")

def disable_torch_init():
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

class Qwen2_5_VL(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if not hasattr(self.visual, "coord_pe_conv"):
            self.visual.add_module("coord_pe_conv", nn.Conv2d(
                in_channels=64*3*2,
                out_channels=1280,
                kernel_size=(14, 14)
            ))
        if not hasattr(self.visual, "coord_pe_mlp"):
            coord_pe_mlp = nn.Sequential(
                nn.Linear(5120, 5120, bias=True),
                nn.GELU(),
                nn.Linear(5120, 5120, bias=True),
                nn.GELU(),
                nn.Linear(5120, 2048, bias=True)
            )
            self.visual.add_module("coord_pe_mlp", coord_pe_mlp)

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map}
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    if use_flash_attn:
        kwargs['_attn_implementation'] = 'flash_attention_2'
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        processor = AutoProcessor.from_pretrained(model_base)
        print('Loading Qwen2-VL from base model...')
        if "Qwen2.5" in model_base:
            model = Qwen2_5_VL.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        print('Loading additional Qwen2-VL weights...')
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model Loaded!!!')
    else:
        processor = AutoProcessor.from_pretrained(model_base)
        if "Qwen2.5" in model_base:
            model = Qwen2_5_VL.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    return processor, model

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

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
        {
            "role": "user",
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

def infer(prompt_text, image_path, depth_path, cam_path):
    conversation = []
    user_content = []
    user_content.append({"type": "image", "image": image_path})
    if prompt_text:
        user_content.append({"type": "text", "text": prompt_text})
    conversation.append({"role": "user", "content": user_content})
    
    full_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversation)
    
    inputs = processor(
        text=[full_prompt],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    coord3d_tensor = None
    if depth_path and cam_path:
        min_pixel = 256 * 28 * 28
        max_pixel = 1280 * 28 * 28
        coord3d = process_depth_and_camera(image_path, depth_path, cam_path, min_pixel, max_pixel)
        coord3d_tensor = torch.tensor(coord3d, dtype=model.dtype, device=device).unsqueeze(0)
    
    # Prepare generation kwargs
    generation_kwargs = {
        "max_new_tokens": generation_args["max_new_tokens"],
        "temperature": generation_args["temperature"],
        "do_sample": (generation_args["temperature"] > 0),
        "repetition_penalty": generation_args["repetition_penalty"],
        "eos_token_id": processor.tokenizer.eos_token_id
    }
    
    # Forward pass through the model
    with torch.inference_mode():
        if coord3d_tensor is not None:
            inputs["coord3d"] = coord3d_tensor
            outputs = model(**inputs)
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=inputs["images"],
                **generation_kwargs
            )
        else:
            output_ids = model.generate(**inputs, **generation_kwargs)
    
    output_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Output:", output_text)

def main(args):
    global processor, model, device, generation_args
    device = args.device
    disable_torch_init()
    use_flash_attn = True
    if args.disable_flash_attention:
        use_flash_attn = False
    replace_qwen2_5_with_mixed_modality_forward()
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        device_map=args.device,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        device=args.device,
        use_flash_attn=use_flash_attn
    )
    model.visual.to(dtype=model.dtype, device=device)
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }
    image_path = "data/coco_3d/train2017/000000191314.jpg"
    depth_path = "data/coco_3d/depth/000000191314_remove_edges.png"
    cam_path = "data/coco_3d/camera_parameters/000000191314.json"
    prompt_text = "<spatial_image>\nWhat element is at position (0.21, 1.87, -0.36)? Provide a single word or phrase."
    infer(prompt_text, image_path, depth_path, cam_path)

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