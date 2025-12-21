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
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature


def prepare_pipeline(unet_config, inference_ckpt_path, seed=1247):

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

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_frames=unet_config.data.num_frames
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    print(f"Loaded checkpoint path: {inference_ckpt_path}")
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(unet_config.model),
        inference_ckpt_path,  # load checkpoint
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

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
    return pipeline, dtype


def main(args):
    unet_config = OmegaConf.load(args.unet_config_path)
    pipeline, dtype = prepare_pipeline(
        unet_config=unet_config,
        inference_ckpt_path=args.inference_ckpt_path,
        seed=args.seed,
    )

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=unet_config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=unet_config.data.resolution,
        height=unet_config.data.resolution,
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
    args = parser.parse_args()

    main(args)
