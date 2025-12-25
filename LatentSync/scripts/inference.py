# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from omegaconf import OmegaConf
import spaces
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature


def trim_video_to_audio_length(video_path: str, audio_length: float):
    """Trim the video to match the length of the audio."""
    from moviepy import VideoFileClip
    import tempfile
    import random

    if 0 < (max_video_length := int(os.getenv("MAX_VIDEO_LENGTH", "0"))):
        audio_length = min(audio_length, max_video_length)
        print(f"Capping audio duration to MAX_VIDEO_LENGTH: {audio_length:.2f}s", flush=True)

    try:
        # Load the video file
        clip = VideoFileClip(video_path)
        video_duration = float(clip.duration)
        if video_duration < audio_length:
            # Audio is longer than video; fall back to full video duration
            print(
                f"Audio ({audio_length:.2f}s) is longer than video ({video_duration:.2f}s); "
                "using full video length.",
                flush=True,
            )
            clip.close()  # Always close the clip when done
            return video_path

        # Compute randomized start/end to match audio_length while staying within video bounds
        max_start = max(0.0, video_duration - audio_length)
        start_time_seconds = random.uniform(0.0, max_start)
        end_time_seconds = start_time_seconds + audio_length

        # Safety clamp: never exceed the clip duration
        end_time_seconds = min(end_time_seconds, video_duration)

        print(
            f"Trimming from {start_time_seconds:.2f}s to {end_time_seconds:.2f}s "
            f"(target audio length: {audio_length:.2f}s)",
            flush=True,
        )

        # Cut the subclip
        trimmed_clip = clip.subclipped(start_time_seconds, end_time_seconds)

        # Write the result to a new file
        # Using 'libx264' codec and 'aac' audio codec for compatibility with MP4
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            trimmed_clip.write_videofile(
                tmp.name,
                codec="libx264",
                audio_codec="aac"
            )

            clip.close()  # Always close the clip when done
            print(f"Video successfully trimmed and saved to {tmp.name}", flush=True)
            return tmp.name

    except Exception as e:  # pylint: disable=broad-except
        print(f"An error occurred: {e}", flush=True)

    return None

def prepare_for_pipeline(unet_config, inference_ckpt_path):
    """Prepare models and scheduler for the lip-sync pipeline."""

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    scheduler = DDIMScheduler.from_pretrained(os.path.join(
        os.environ["LATENTSYNC_DIR"], "configs/"
    ))

    if unet_config.model.cross_attention_dim == 768:
        whisper_model_path = os.path.join(
            os.environ["LATENTSYNC_DIR"], "checkpoints/whisper/small.pt"
        )
    elif unet_config.model.cross_attention_dim == 384:
        whisper_model_path = os.path.join(
            os.environ["LATENTSYNC_DIR"], "checkpoints/whisper/tiny.pt"
        )
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    print(f"Loaded checkpoint path: {inference_ckpt_path}")
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(unet_config.model),
        inference_ckpt_path,  # load checkpoint
        device="cpu",
    )

    unet = unet.to(dtype=dtype)
    return dtype, whisper_model_path, unet, scheduler

@spaces.GPU(duration=int(os.getenv("SPACES_GPU_TIMEOUT", "120")))
def run_pipeline(
    audio_model_path,
    unet_config,
    unet,
    scheduler,
    video_path,
    audio_path,
    audio_length,
    video_out_path,
    num_inference_steps=20,
    guidance_scale=1.0,
    weight_dtype=torch.float32,
    width=256,
    height=256,
    seed=1247
):
    """Run the lip-sync pipeline."""
    audio_encoder = Audio2Feature(
        model_path=audio_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_frames=unet_config.data.num_frames
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=weight_dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    # set xformers
    if is_xformers_available() and torch.cuda.is_available():
        unet.enable_xformers_memory_efficient_attention()

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    if seed != -1:
        set_seed(seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")
    print(f"Input video path: {video_path}")
    print(f"Input audio path: {audio_path}")

    trimmed_video_path = trim_video_to_audio_length(
        video_path, audio_length
    ) if audio_length else video_path
    print(f"Trimmed video path: {trimmed_video_path}")

    pipeline(
        video_path=trimmed_video_path,
        audio_path=audio_path,
        video_out_path=video_out_path,
        video_mask_path=video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=unet_config.data.num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        weight_dtype=weight_dtype,
        width=width,
        height=height,
    )


def main(args):
    unet_config = OmegaConf.load(args.unet_config_path)
    weight_dtype, audio_model_path, unet, scheduler = prepare_for_pipeline(
        unet_config=unet_config,
        inference_ckpt_path=args.inference_ckpt_path,
    )
    run_pipeline(
        audio_model_path=audio_model_path,
        unet_config=unet_config,
        unet=unet,
        scheduler=scheduler,
        video_path=args.video_path,
        audio_path=args.audio_path,
        audio_length=None,
        video_out_path=args.video_out_path,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=weight_dtype,
        width=unet_config.data.resolution,
        height=unet_config.data.resolution,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parsed_args = parser.parse_args()

    main(parsed_args)
