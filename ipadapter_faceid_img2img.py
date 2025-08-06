"""
IPAdapter FaceID Image-to-Image Example

This example demonstrates FaceID-conditioned image-to-image generation
using multiple portrait inputs for face consistency across different styles.
"""

import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from PIL import Image
import os

# Import the enhanced IPAdapter
from ip_adapter.ip_adapter import IPAdapter

def setup_img2img_pipeline(model_path="runwayml/stable-diffusion-v1-5", device="cuda"):
    """Setup the Stable Diffusion image-to-image pipeline."""
    print("setup_img2img_pipeline: Loading Stable Diffusion img2img pipeline...")
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    
    print("setup_img2img_pipeline: Pipeline loaded successfully")
    return pipe

def generate_faceid_img2img(
    ip_adapter,
    face_image_path,
    init_image_path,
    prompt,
    negative_prompt="blurry, low quality, distorted",
    strength=0.75,
    num_inference_steps=30,
    guidance_scale=7.5,
    faceid_v2_weight=2.0,
    seed=42,
    output_path="output_faceid_img2img.png"
):
    """Generate an image using FaceID conditioning with img2img."""
    print(f"generate_faceid_img2img: Loading face image from {face_image_path}")
    print(f"generate_faceid_img2img: Loading init image from {init_image_path}")
    
    # Load and prepare images
    face_image = Image.open(face_image_path).convert("RGB")
    init_image = Image.open(init_image_path).convert("RGB")
    
    # Resize to appropriate dimensions
    face_image = face_image.resize((512, 512))
    init_image = init_image.resize((512, 512))
    
    print(f"generate_faceid_img2img: Generating embeddings for prompt: '{prompt}'")
    
    # Generate prompt embeddings with FaceID conditioning
    prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
        images=face_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        faceid_v2_weight=faceid_v2_weight if ip_adapter.is_faceidv2 else 1.0
    )
    
    print(f"generate_faceid_img2img: Prompt embeddings shape: {prompt_embeds.shape}")
    print(f"generate_faceid_img2img: Negative embeddings shape: {negative_prompt_embeds.shape}")
    
    # Generate image
    print("generate_faceid_img2img: Generating image with img2img...")
    generator = torch.Generator(device=ip_adapter.device).manual_seed(seed)
    
    with torch.no_grad():
        result = ip_adapter.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    
    # Save result
    result.images[0].save(output_path)
    print(f"generate_faceid_img2img: Image saved to {output_path}")
    
    return result.images[0]

def example_faceid_img2img_basic():
    """Basic FaceID img2img example."""
    print("\\n=== Basic FaceID Image-to-Image Example ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup img2img pipeline
    pipe = setup_img2img_pipeline(device=device)
    
    # Load FaceID adapter (update paths as needed)
    ip_adapter = IPAdapter(
        pipe=pipe,
        ipadapter_ckpt_path="models/models--h94--IP-Adapter-FaceID/snapshots/43907e6f44d079bf1a9102d9a6e56aef7a219bae/ip-adapter-faceid_sd15.bin",
        image_encoder_path="models/clip_encoder/sd15/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder",
        device=device,
        insightface_model_name="buffalo_l"
    )
    
    print(f"example_faceid_img2img_basic: Model features:")
    print(f"  - FaceID: {ip_adapter.is_faceid}")
    print(f"  - Plus variant: {ip_adapter.is_plus}")
    print(f"  - FaceID v2: {ip_adapter.is_faceidv2}")
    
    # Generate image using both portraits
    generate_faceid_img2img(
        ip_adapter=ip_adapter,
        face_image_path="assets/input_portrait.jpg",  # Face reference
        init_image_path="assets/input_portrait2.jpg",  # Image to transform
        prompt="professional business portrait, formal attire, studio lighting",
        strength=0.7,
        output_path="test_outputs/faceid_img2img_basic.png"
    )

def example_faceid_img2img_style_transfer():
    """FaceID img2img style transfer examples."""
    print("\\n=== FaceID Image-to-Image Style Transfer ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup pipeline
    pipe = setup_img2img_pipeline(device=device)
    
    # Load FaceID adapter
    ip_adapter = IPAdapter(
        pipe=pipe,
        ipadapter_ckpt_path="models/models--h94--IP-Adapter-FaceID/snapshots/43907e6f44d079bf1a9102d9a6e56aef7a219bae/ip-adapter-faceid_sd15.bin",
        image_encoder_path="models/clip_encoder/sd15/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder",
        device=device,
        insightface_model_name="buffalo_l"
    )
    
    # Different style transformations
    style_configs = [
        {
            "prompt": "Renaissance oil painting style, classical art, detailed brushwork",
            "strength": 0.8,
            "seed": 100,
            "output": "test_outputs/faceid_img2img_renaissance.png"
        },
        {
            "prompt": "anime style portrait, manga character, cel shading, vibrant colors",
            "strength": 0.75,
            "seed": 200,
            "output": "test_outputs/faceid_img2img_anime.png"
        },
        {
            "prompt": "cyberpunk portrait, neon lighting, futuristic, high tech",
            "strength": 0.7,
            "seed": 300,
            "output": "test_outputs/faceid_img2img_cyberpunk.png"
        },
        {
            "prompt": "watercolor painting, soft brushstrokes, artistic, dreamy",
            "strength": 0.8,
            "seed": 400,
            "output": "test_outputs/faceid_img2img_watercolor.png"
        }
    ]
    
    for i, config in enumerate(style_configs):
        print(f"\\nGenerating style {i+1}/4: {config['prompt'][:30]}...")
        generate_faceid_img2img(
            ip_adapter=ip_adapter,
            face_image_path="assets/input_portrait.jpg",  # Face reference
            init_image_path="assets/input_portrait2.jpg",  # Base image
            prompt=config["prompt"],
            strength=config["strength"],
            seed=config["seed"],
            output_path=config["output"]
        )

def example_cross_portrait_generation():
    """Example using different portrait combinations."""
    print("\\n=== Cross-Portrait FaceID Generation ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup pipeline
    pipe = setup_img2img_pipeline(device=device)
    
    # Load FaceID adapter
    ip_adapter = IPAdapter(
        pipe=pipe,
        ipadapter_ckpt_path="models/models--h94--IP-Adapter-FaceID/snapshots/43907e6f44d079bf1a9102d9a6e56aef7a219bae/ip-adapter-faceid_sd15.bin",
        image_encoder_path="models/clip_encoder/sd15/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder",
        device=device,
        insightface_model_name="buffalo_l"
    )
    
    # Cross combinations
    combinations = [
        {
            "face_ref": "assets/input_portrait.jpg",
            "init_img": "assets/input_portrait2.jpg", 
            "prompt": "professional headshot, business attire, confident expression",
            "output": "test_outputs/faceid_portrait1_to_portrait2.png"
        },
        {
            "face_ref": "assets/input_portrait2.jpg",
            "init_img": "assets/input_portrait.jpg",
            "prompt": "casual portrait, relaxed expression, natural lighting", 
            "output": "test_outputs/faceid_portrait2_to_portrait1.png"
        }
    ]
    
    for i, combo in enumerate(combinations):
        print(f"\\nGenerating combination {i+1}/2...")
        if os.path.exists(combo["face_ref"]) and os.path.exists(combo["init_img"]):
            generate_faceid_img2img(
                ip_adapter=ip_adapter,
                face_image_path=combo["face_ref"],
                init_image_path=combo["init_img"],
                prompt=combo["prompt"],
                strength=0.75,
                seed=500 + i,
                output_path=combo["output"]
            )
        else:
            print(f"Skipping combination {i+1} - missing input files")

def main():
    """Main function to run img2img examples."""
    print("🎨 IPAdapter FaceID Image-to-Image Examples")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("test_outputs", exist_ok=True)
    
    # Check for input images
    required_images = ["assets/input_portrait.jpg", "assets/input_portrait2.jpg"]
    missing_images = [img for img in required_images if not os.path.exists(img)]
    
    if missing_images:
        print("⚠️  Missing required input images:")
        for img in missing_images:
            print(f"   - {img}")
        print("\\nPlease ensure both portrait images are available in the assets directory.")
        return
    
    try:
        # Run examples
        example_faceid_img2img_basic()
        example_faceid_img2img_style_transfer()
        example_cross_portrait_generation()
        
        print("\\n✅ All img2img examples completed successfully!")
        print("Check the 'test_outputs' directory for generated images.")
        print("\\n📋 Generated files:")
        print("  - faceid_img2img_basic.png")
        print("  - faceid_img2img_renaissance.png")
        print("  - faceid_img2img_anime.png") 
        print("  - faceid_img2img_cyberpunk.png")
        print("  - faceid_img2img_watercolor.png")
        print("  - faceid_portrait1_to_portrait2.png")
        print("  - faceid_portrait2_to_portrait1.png")
        
    except Exception as e:
        print(f"\\n❌ Error running img2img examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()