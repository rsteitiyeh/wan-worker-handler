import os, base64, tempfile, traceback, torch
import runpod
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
NEG = ("color saturated, overexposed, static, blurry, subtitles, worst quality, low quality, "
       "JPEG artifacts, ugly, deformed, extra fingers, bad hands, bad face, watermark, text, cartoon")
_pipe = None
def get_pipe():
    global _pipe
    if _pipe is None:
        vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
        p = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
        p.enable_model_cpu_offload()
        try: p.vae.enable_tiling()
        except Exception: pass
        _pipe = p
    return _pipe

def handler(event):
    try:
        inp = event.get("input", {})
        prompt = inp.get("prompt", "")
        frames = int(inp.get("frames", 121)); steps = int(inp.get("steps", 35))
        seed = int(inp.get("seed", 7)); name = inp.get("name", "clip")
        pipe = get_pipe()
        out = pipe(prompt=prompt, negative_prompt=NEG, height=704, width=1280,
                   num_frames=frames, guidance_scale=5.0, num_inference_steps=steps,
                   generator=torch.Generator("cpu").manual_seed(seed)).frames[0]
        fp = tempfile.mktemp(suffix=".mp4")
        export_to_video(out, fp, fps=24)
        with open(fp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return {"name": name, "mp4_b64": b64, "frames": frames, "steps": steps}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-1500:]}

runpod.serverless.start({"handler": handler})
