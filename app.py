import streamlit as st
import torch
import random
from diffusers import StableDiffusionXLPipeline

st.set_page_config(
    page_title="ImagineAI | AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

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
    "Fast": {"steps": 20, "guidance": 6.5},
    "Balanced": {"steps": 30, "guidance": 7.5},
    "High Quality": {"steps": 40, "guidance": 8.0}
}

ASPECT_RATIOS = {
    "Square (1:1)": (768, 768),
    "Portrait (2:3)": (512, 768),
    "Landscape (3:2)": (768, 512),
    "Wide (16:9)": (768, 432)
}

EXAMPLE_PROMPTS = [
    "A futuristic Islamabad city at sunset with flying cars and modern architecture",
    "A luxury Pakistani wedding portrait of a bride and groom wearing traditional clothes, highly detailed",
    "A futuristic humanoid robot working in a modern technology laboratory",
    "A beautiful mountain lake surrounded by snowy mountains at sunrise",
    "A modern glass house in the middle of a peaceful forest during golden hour",
    "An astronaut standing on a colorful alien planet with two moons in the sky"
]

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True
    )

    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe, device

def generate_image(
    pipe,
    device,
    prompt,
    negative_prompt,
    style,
    aspect_ratio,
    quality,
    seed
):
    if not prompt or not prompt.strip():
        raise ValueError("Please enter a description for your image.")

    if seed == -1:
        seed = random.randint(0, 2147483647)

    width, height = ASPECT_RATIOS[aspect_ratio]
    settings = QUALITY_SETTINGS[quality]

    final_prompt = prompt.strip()

    if STYLE_PROMPTS[style]:
        final_prompt = f"{final_prompt}, {STYLE_PROMPTS[style]}"

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

    generator = torch.Generator(device=device).manual_seed(int(seed))

    image = pipe(
        prompt=final_prompt,
        negative_prompt=final_negative_prompt,
        width=width,
        height=height,
        num_inference_steps=settings["steps"],
        guidance_scale=settings["guidance"],
        generator=generator
    ).images[0]

    return image, int(seed), width, height

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #111827, #172554);
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 35px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🎨 ImagineAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Transform your imagination into stunning AI-generated images</div>',
    unsafe_allow_html=True
)

left_column, right_column = st.columns(2)

with left_column:
    st.markdown(
        '<div class="section-title">✨ Create Your Image</div>',
        unsafe_allow_html=True
    )

    prompt = st.text_area(
        "Describe your image",
        placeholder="Describe anything you can imagine...",
        height=160
    )

    style = st.selectbox(
        "🎨 Image Style",
        list(STYLE_PROMPTS.keys()),
        index=1
    )

    aspect_ratio = st.selectbox(
        "📐 Aspect Ratio",
        list(ASPECT_RATIOS.keys())
    )

    quality = st.radio(
        "⚡ Generation Quality",
        list(QUALITY_SETTINGS.keys()),
        index=1
    )

    with st.expander("⚙️ Advanced Settings"):
        negative_prompt = st.text_area(
            "Negative Prompt",
            placeholder="Things you do not want in the image..."
        )

        seed = st.number_input(
            "Seed (-1 = Random)",
            value=-1,
            step=1
        )

    st.markdown("### 💡 Inspiration")

    selected_example = st.selectbox(
        "Choose an idea",
        ["Select an example"] + EXAMPLE_PROMPTS
    )

    if selected_example != "Select an example":
        st.info(selected_example)

        if st.button("Use This Idea"):
            st.session_state["selected_prompt"] = selected_example
            st.rerun()

    generate_button = st.button(
        "✨ Generate Image",
        use_container_width=True,
        type="primary"
    )

with right_column:
    st.markdown(
        '<div class="section-title">🖼️ Your Creation</div>',
        unsafe_allow_html=True
    )

    if generate_button:
        try:
            with st.spinner("🎨 ImagineAI is creating your image..."):
                pipe, device = load_model()

                image, used_seed, width, height = generate_image(
                    pipe,
                    device,
                    prompt,
                    negative_prompt,
                    style,
                    aspect_ratio,
                    quality,
                    int(seed)
                )

            st.image(
                image,
                caption=f"{style} • {width} × {height} • Seed: {used_seed}",
                use_container_width=True
            )

            st.success("Image generated successfully!")

        except Exception as e:
            st.error(f"Generation failed: {str(e)}")

    else:
        st.info("Your generated image will appear here.")
