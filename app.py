import torch
import gradio as gr
import random
from diffusers import StableDiffusionXLPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading ImagineAI on {device}...")

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    use_safetensors=True
)

pipe = pipe.to(device)

if device == "cuda":
    pipe.enable_attention_slicing()

STYLE_PROMPTS = {
    "None": "",
    "Photorealistic": "professional photography, photorealistic, realistic skin texture, natural lighting, sharp focus, highly detailed",
    "Cinematic": "cinematic composition, dramatic lighting, movie scene, volumetric lighting, highly detailed, sharp focus",
    "Digital Art": "professional digital artwork, highly detailed, beautiful composition, concept art",
    "Anime": "anime style, highly detailed anime artwork, beautiful colors, clean line art",
    "Fantasy": "fantasy art, magical atmosphere, epic composition, highly detailed, cinematic lighting",
    "3D Render": "high quality 3D render, detailed textures, realistic lighting, professional 3D artwork",
    "Watercolor": "beautiful watercolor painting, artistic brush strokes, detailed composition"
}

QUALITY_SETTINGS = {
    "Fast": {
        "steps": 20,
        "guidance": 6.5
    },
    "Balanced": {
        "steps": 30,
        "guidance": 7.5
    },
    "High Quality": {
        "steps": 40,
        "guidance": 8.0
    }
}

ASPECT_RATIOS = {
    "Square (1:1)": (1024, 1024),
    "Portrait (2:3)": (768, 1152),
    "Landscape (3:2)": (1152, 768),
    "Wide (16:9)": (1152, 648)
}

EXAMPLE_PROMPTS = [
    "A futuristic Islamabad city at sunset with flying cars and modern architecture",
    "A luxury Pakistani wedding portrait of a bride and groom wearing traditional clothes, highly detailed",
    "A futuristic humanoid robot working in a modern technology laboratory",
    "A beautiful mountain lake surrounded by snowy mountains at sunrise",
    "A modern glass house in the middle of a peaceful forest during golden hour",
    "An astronaut standing on a colorful alien planet with two moons in the sky"
]

def generate_image(prompt, negative_prompt, style, aspect_ratio, quality, seed):
    try:
        if not prompt or not prompt.strip():
            raise gr.Error("Please enter a description for your image.")

        if int(seed) == -1:
            seed = random.randint(0, 2147483647)

        seed = int(seed)

        width, height = ASPECT_RATIOS[aspect_ratio]

        settings = QUALITY_SETTINGS[quality]
        steps = settings["steps"]
        guidance = settings["guidance"]

        final_prompt = prompt.strip()
        style_prompt = STYLE_PROMPTS[style]

        if style_prompt:
            final_prompt = f"{final_prompt}, {style_prompt}"

        default_negative_prompt = (
            "blurry, blur, low quality, low resolution, pixelated, "
            "out of focus, distorted, deformed, bad anatomy, bad hands, "
            "extra fingers, missing fingers, extra limbs, duplicate, ugly, "
            "poorly drawn, cropped, watermark, text, logo, oversaturated"
        )

        final_negative_prompt = default_negative_prompt

        if negative_prompt and negative_prompt.strip():
            final_negative_prompt = (
                f"{default_negative_prompt}, {negative_prompt.strip()}"
            )

        generator = torch.Generator(device=device).manual_seed(seed)

        image = pipe(
            prompt=final_prompt,
            negative_prompt=final_negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]

        status = (
            f"### ✅ Image Generated Successfully!\n"
            f"**Style:** {style} | "
            f"**Quality:** {quality} | "
            f"**Size:** {width} × {height} | "
            f"**Seed:** {seed}"
        )

        return image, seed, status

    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

def random_prompt():
    return random.choice(EXAMPLE_PROMPTS)

def clear_all():
    return (
        "",
        "",
        "Photorealistic",
        "Square (1:1)",
        "Balanced",
        -1,
        None,
        -1,
        "### ✨ Ready to create something amazing!"
    )

css = """
body {
    background: #0b1020 !important;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    padding-top: 25px !important;
}

#main-container {
    background: linear-gradient(145deg, #111827, #0f172a);
    border: 1px solid #293548;
    border-radius: 24px;
    padding: 24px;
}

#hero {
    text-align: center;
    padding: 30px 20px 35px 20px;
}

#hero-title {
    font-size: 52px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#hero-subtitle {
    font-size: 18px;
    color: #94a3b8;
    margin-top: 10px;
}

#section-card {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 20px;
}

.section-heading {
    font-size: 20px;
    font-weight: 700;
    color: white;
    margin-bottom: 18px;
}

#generate-button {
    min-height: 55px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
}

#footer {
    text-align: center;
    color: #64748b;
    padding: 25px 10px 5px 10px;
}
"""

with gr.Blocks(
    title="ImagineAI | Professional AI Image Generator",
    theme=gr.themes.Base(),
    css=css
) as demo:

    with gr.Column(elem_id="main-container"):

        gr.HTML("""
        <div id="hero">
            <div id="hero-title">🎨 ImagineAI</div>
            <div id="hero-subtitle">
                Transform your imagination into stunning AI-generated images
            </div>
        </div>
        """)

        with gr.Row():

            with gr.Column(scale=1, elem_id="section-card"):

                gr.HTML("""
                <div class="section-heading">✨ Create Your Image</div>
                """)

                prompt = gr.Textbox(
                    label="Describe your image",
                    placeholder="Describe anything you can imagine...",
                    lines=6
                )

                with gr.Row():
                    random_button = gr.Button(
                        "🎲 Random Idea",
                        size="sm"
                    )

                    clear_button = gr.Button(
                        "🗑️ Clear",
                        size="sm"
                    )

                gr.Markdown(
                    "💡 **Pro Tip:** Describe the subject, environment, lighting, style and important details for better results."
                )

                style = gr.Dropdown(
                    choices=list(STYLE_PROMPTS.keys()),
                    value="Photorealistic",
                    label="🎨 Image Style"
                )

                aspect_ratio = gr.Dropdown(
                    choices=list(ASPECT_RATIOS.keys()),
                    value="Square (1:1)",
                    label="📐 Aspect Ratio"
                )

                quality = gr.Radio(
                    choices=list(QUALITY_SETTINGS.keys()),
                    value="Balanced",
                    label="⚡ Generation Quality"
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False):

                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Things you do not want in the image..."
                    )

                    seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="Seed",
                        info="Use -1 for a random image"
                    )

                generate_button = gr.Button(
                    "✨ Generate Image",
                    variant="primary",
                    size="lg",
                    elem_id="generate-button"
                )

            with gr.Column(scale=1, elem_id="section-card"):

                gr.HTML("""
                <div class="section-heading">🖼️ Your Creation</div>
                """)

                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=550
                )

                output_seed = gr.Number(
                    label="Seed Used",
                    precision=0
                )

                status = gr.Markdown(
                    "### ✨ Ready to create something amazing!"
                )

        gr.Markdown("## 💡 Inspiration Gallery")

        gr.Examples(
            examples=[
                [EXAMPLE_PROMPTS[0], "", "Cinematic", "Square (1:1)", "Balanced", -1],
                [EXAMPLE_PROMPTS[1], "", "Photorealistic", "Portrait (2:3)", "High Quality", -1],
                [EXAMPLE_PROMPTS[2], "", "3D Render", "Square (1:1)", "Balanced", -1],
                [EXAMPLE_PROMPTS[3], "", "Photorealistic", "Landscape (3:2)", "High Quality", -1],
                [EXAMPLE_PROMPTS[4], "", "Photorealistic", "Landscape (3:2)", "Balanced", -1],
                [EXAMPLE_PROMPTS[5], "", "Fantasy", "Wide (16:9)", "High Quality", -1]
            ],
            inputs=[
                prompt,
                negative_prompt,
                style,
                aspect_ratio,
                quality,
                seed
            ],
            label="Click an idea to use it"
        )

        gr.HTML("""
        <div id="footer">
            <b>ImagineAI</b> • AI-Powered Image Generation • Built with Python, Diffusers & Gradio
        </div>
        """)

    generate_button.click(
        fn=generate_image,
        inputs=[
            prompt,
            negative_prompt,
            style,
            aspect_ratio,
            quality,
            seed
        ],
        outputs=[
            output_image,
            output_seed,
            status
        ]
    )

    random_button.click(
        fn=random_prompt,
        inputs=[],
        outputs=prompt
    )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            prompt,
            negative_prompt,
            style,
            aspect_ratio,
            quality,
            seed,
            output_image,
            output_seed,
            status
        ]
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
