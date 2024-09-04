import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from huggingface_hub import login

def load_model(model_name):
    print(f"Loading model: {model_name}...")
    try:
        model = StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        model = model.to("cuda")
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Failed to load model {model_name}. Error: {str(e)}")
        model = None
    return model

def prepare_images(input_image_path):
    print("Preparing images...")
    input_image = Image.open(input_image_path).convert("RGB")
    input_width, input_height = input_image.size

    # Create 42:9 canvas
    new_width = input_width * 2
    new_image = Image.new("RGB", (new_width, input_height))
    new_image.paste(input_image, (input_width // 2, 0))

    # Create mask: white on sides, black in center
    mask = Image.new("L", (new_width, input_height), 255)  # Start with white
    mask.paste(0, (input_width // 2, 0, input_width * 3 // 2, input_height))  # Black in center

    print("Images prepared successfully.")
    return new_image, mask, input_width

def split_image(image, original_width):
    left = image.crop((0, 0, original_width // 2, image.height))
    right = image.crop((original_width * 3 // 2, 0, image.width, image.height))
    return left, right

def perform_outpainting(model, image, mask, prompt, negative_prompt, original_width):
    print("Starting outpainting...")

    # SDXL prefers larger input sizes
    target_size = (1024, int(1024 * (image.height / image.width)))

    resized_image = image.resize(target_size, Image.LANCZOS)
    resized_mask = mask.resize(target_size, Image.LANCZOS)

    # Convert mask to numpy array and normalize
    mask_array = np.array(resized_mask)
    mask_array = mask_array.astype(np.float32) / 255.0

    # Perform outpainting
    output = model(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=resized_image,
        mask_image=mask_array,
        num_inference_steps=50,
        guidance_scale=7.5,
        strength=1.0
    ).images[0]

    # Resize output back to original aspect ratio
    final_width = original_width * 2  # 42:9 aspect ratio
    final_height = int(final_width * (image.height / image.width))
    output = output.resize((final_width, final_height), Image.LANCZOS)

    # Split the output image
    left, right = split_image(output, original_width)

    print("Outpainting completed.")
    return left, right

def combine_images(original, left, right):
    new_width = original.width * 2
    result = Image.new("RGB", (new_width, original.height))
    result.paste(left, (0, 0))
    result.paste(original, (original.width // 2, 0))
    result.paste(right, (original.width * 3 // 2, 0))
    return result

if __name__ == "__main__":
    # Authenticate if necessary
    try:
        login(token="your_huggingface_token_here")  # Replace with your actual token
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
    
    model_name = "stabilityai/stable-diffusion-xl-1.0-inpainting-0.1"
    model = load_model(model_name)
    
    # Fallback to Stable Diffusion 2.1 inpainting if SDXL fails
    if model is None:
        model_name = "stabilityai/stable-diffusion-2-inpainting"
        model = load_model(model_name)
        if model is None:
            print("Failed to load any model. Exiting.")
            exit(1)
    
    input_image_path = "img.png"  # Replace with your input image path
    output_image_path = "outpainted_image.png"
    
    image, mask, original_width = prepare_images(input_image_path)
    original_image = Image.open(input_image_path).convert("RGB")
    
    prompt = ("Extend this image seamlessly. Match the exact style, colors, and content of the original. "
              "Ensure perfect continuity and consistent lighting. Create a natural continuation "
              "of the scene that blends flawlessly with the existing image.")
    
    negative_prompt = ("discontinuity, mismatch, inconsistent style, different colors, "
                       "unrealistic extension, poor quality, low resolution, blurry, distorted")
    
    left, right = perform_outpainting(model, image, mask, prompt, negative_prompt, original_width)
    result = combine_images(original_image, left, right)
    result.save(output_image_path)
    print(f"Final image saved as {output_image_path}")
