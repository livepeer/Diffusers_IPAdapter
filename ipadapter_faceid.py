"""
IPAdapter FaceID Example Implementation

This example demonstrates how to use the enhanced IPAdapter with FaceID support
for face-conditioned image generation.
"""

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import os

# Import the enhanced IPAdapter
from ip_adapter.ip_adapter import IPAdapter

def setup_pipeline(model_path="runwayml/stable-diffusion-v1-5", device="cuda"):
    """Setup the Stable Diffusion pipeline."""
    print("setup_pipeline: Loading Stable Diffusion pipeline...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    
    print("setup_pipeline: Pipeline loaded successfully")
    return pipe

def load_faceid_adapter(
    pipe, 
    faceid_model_path, 
    image_encoder_path,
    device="cuda",
    insightface_model="buffalo_l"
):
    """Load IPAdapter with FaceID support."""
    print(f"load_faceid_adapter: Loading IPAdapter FaceID model from {faceid_model_path}")
    
    ip_adapter = IPAdapter(
        pipe=pipe,
        ipadapter_ckpt_path=faceid_model_path,
        image_encoder_path=image_encoder_path,
        device=device,
        insightface_model_name=insightface_model
    )
    
    print("load_faceid_adapter: IPAdapter loaded with the following features:")
    print(f"  - FaceID: {ip_adapter.is_faceid}")
    print(f"  - Plus variant: {ip_adapter.is_plus}")
    print(f"  - FaceID v2: {ip_adapter.is_faceidv2}")
    print(f"  - SDXL: {ip_adapter.is_sdxl}")
    print(f"  - Portrait unnorm: {ip_adapter.is_portrait_unnorm}")
    
    return ip_adapter

def generate_faceid_image(
    ip_adapter,
    face_image_path,
    prompt,
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=30,
    guidance_scale=7.5,
    faceid_v2_weight=2.0,
    seed=42,
    output_path="output_faceid.png"
):
    """Generate an image using FaceID conditioning."""
    print(f"generate_faceid_image: Loading face image from {face_image_path}")
    
    # Load and prepare face image
    face_image = Image.open(face_image_path).convert("RGB")
    
    # Resize to appropriate dimensions
    face_image = face_image.resize((512, 512))
    
    print(f"generate_faceid_image: Generating embeddings for prompt: '{prompt}'")
    
    # Generate prompt embeddings with FaceID conditioning
    prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
        images=face_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        faceid_v2_weight=faceid_v2_weight if ip_adapter.is_faceidv2 else 1.0
    )
    
    print(f"generate_faceid_image: Prompt embeddings shape: {prompt_embeds.shape}")
    print(f"generate_faceid_image: Negative embeddings shape: {negative_prompt_embeds.shape}")
    
    # Generate image
    print("generate_faceid_image: Generating image...")
    generator = torch.Generator(device=ip_adapter.device).manual_seed(seed)
    
    with torch.no_grad():
        result = ip_adapter.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=512,
            width=512
        )
    
    # Save result
    result.images[0].save(output_path)
    print(f"generate_faceid_image: Image saved to {output_path}")
    
    return result.images[0]

def example_basic_faceid():
    """Basic FaceID example with a standard FaceID model."""
    print("\\n=== Basic FaceID Example ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup pipeline
    pipe = setup_pipeline(device=device)
    
    # Load FaceID adapter (update paths as needed)
    ip_adapter = load_faceid_adapter(
        pipe=pipe,
        faceid_model_path="path/to/ip-adapter-faceid_sd15.bin",  # Update this path
        image_encoder_path="path/to/image_encoder",  # Update this path
        device=device
    )
    
    # Generate image
    generate_faceid_image(
        ip_adapter=ip_adapter,
        face_image_path="path/to/face_image.jpg",  # Update this path
        prompt="professional headshot, business attire, studio lighting",
        output_path="output/basic_faceid.png"
    )

def example_faceid_plus():
    """FaceID Plus example with advanced features."""
    print("\\n=== FaceID Plus Example ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup pipeline
    pipe = setup_pipeline(device=device)
    
    # Load FaceID Plus adapter
    ip_adapter = load_faceid_adapter(
        pipe=pipe,
        faceid_model_path="path/to/ip-adapter-faceid-plusv2_sd15.bin",  # Update this path
        image_encoder_path="path/to/image_encoder",  # Update this path
        device=device
    )
    
    # Generate multiple variations
    prompts = [
        "portrait in renaissance style, oil painting",
        "cyberpunk character, neon lighting, futuristic",
        "medieval knight, armor, heroic pose",
        "modern fashion portrait, elegant, high fashion"
    ]
    
    for i, prompt in enumerate(prompts):
        generate_faceid_image(
            ip_adapter=ip_adapter,
            face_image_path="path/to/face_image.jpg",  # Update this path
            prompt=prompt,
            faceid_v2_weight=2.0,  # Higher weight for FaceID v2
            seed=42 + i,
            output_path=f"output/faceid_plus_variation_{i+1}.png"
        )

def example_batch_generation():
    """Example of batch generation with multiple face images."""
    print("\\n=== Batch FaceID Generation Example ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup pipeline
    pipe = setup_pipeline(device=device)
    
    # Load FaceID adapter
    ip_adapter = load_faceid_adapter(
        pipe=pipe,
        faceid_model_path="path/to/ip-adapter-faceid-plusv2_sd15.bin",  # Update this path
        image_encoder_path="path/to/image_encoder",  # Update this path
        device=device
    )
    
    # Multiple face images
    face_images = [
        "path/to/face1.jpg",  # Update these paths
        "path/to/face2.jpg",
        "path/to/face3.jpg"
    ]
    
    base_prompt = "professional portrait, studio lighting, high quality"
    
    for i, face_path in enumerate(face_images):
        if os.path.exists(face_path):
            generate_faceid_image(
                ip_adapter=ip_adapter,
                face_image_path=face_path,
                prompt=base_prompt,
                seed=100 + i,
                output_path=f"output/batch_faceid_{i+1}.png"
            )
        else:
            print(f"example_batch_generation: Face image not found: {face_path}")

def example_style_transfer():
    """Example of style transfer with FaceID portrait models."""
    print("\\n=== FaceID Style Transfer Example ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup pipeline
    pipe = setup_pipeline(device=device)
    
    # Load FaceID Portrait adapter (specialized for style transfer)
    ip_adapter = load_faceid_adapter(
        pipe=pipe,
        faceid_model_path="path/to/ip-adapter-faceid-portrait-v11_sd15.bin",  # Update this path
        image_encoder_path="path/to/image_encoder",  # Update this path
        device=device
    )
    
    # Style transfer prompts
    styles = [
        "in the style of Van Gogh, impressionist painting",
        "anime style, manga character, detailed",
        "photorealistic, hyperdetailed, 8k resolution",
        "watercolor painting, soft brushstrokes, artistic"
    ]
    
    for i, style in enumerate(styles):
        generate_faceid_image(
            ip_adapter=ip_adapter,
            face_image_path="path/to/face_image.jpg",  # Update this path
            prompt=f"portrait {style}",
            guidance_scale=8.0,  # Higher guidance for style transfer
            seed=200 + i,
            output_path=f"output/style_transfer_{i+1}.png"
        )

def main():
    """Main function to run examples."""
    print("🎨 IPAdapter FaceID Examples")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Note: Update the file paths in the examples before running
    print("⚠️  Note: Update the model and image paths in the examples before running!")
    print("   - FaceID model paths (*.bin files)")
    print("   - Image encoder path")
    print("   - Face image paths (*.jpg files)")
    print()
    
    try:
        # Run examples (uncomment as needed)
        
        # example_basic_faceid()
        # example_faceid_plus()
        # example_batch_generation()
        # example_style_transfer()
        
        print("\\n✅ All examples completed successfully!")
        print("Check the 'output' directory for generated images.")
        
    except Exception as e:
        print(f"\\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()