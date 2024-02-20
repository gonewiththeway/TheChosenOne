from diffusers import DiffusionPipeline
import torch
import argparse
import yaml
import os

def config_2_args(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="Generate args from yaml")
    for key, value in yaml_data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
    return args

args = config_2_args("config/princess.yaml")

model_path = os.path.join(args.output_dir, args.character_name)
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.load_lora_weights(os.path.join(model_path, f"checkpoint-{args.checkpointing_steps * args.num_train_epochs}"))

output_folder = f"./inference_results/{args.character_name}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

while True:
    prompt_postfix = input("Enter the postfix for photo_man or type 'exit' to quit: ")
    if prompt_postfix.lower() == 'exit':
        break

    image_postfix = prompt_postfix.replace(" ", "_")
    prompt = f"A photo of {args.placeholder_token} {prompt_postfix}."
    print(prompt)

    image = pipe(prompt, num_inference_steps=35, guidance_scale=7.5).images[0]
    image.save(os.path.join(output_folder, f"{args.character_name}_{image_postfix}.png"))
    print(f"Image saved as {args.character_name}_{image_postfix}.png in {output_folder}")
