import torch

from PIL.Image import Image

from core.models.base import (
    HuggingFaceModel,
    Condition,
    ModelOutput,
    PredictionType,
    save_png,
)


class StableDiffusionModel(HuggingFaceModel[Image]):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe.safety_checker = None

    def get_network(self):
        return self.pipe.unet

    def encode_condition(self, **kwargs) -> Condition:

        prompts: list[str] = kwargs["prompts"]
        guidance_scale: float = kwargs.get("guidance_scale", 7.5)

        # 1. Check inputs
        self.pipe._guidance_scale = guidance_scale
        self.pipe._guidance_rescale = None
        self.pipe._clip_skip = None
        self.pipe._cross_attention_kwargs = None
        self.pipe._interrupt = False

        # 2. Define call parameters
        device = self.pipe._execution_device

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompts, device, 1, True
        )

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "guidance_scale": guidance_scale,
        }

    def get_prior(self, seeds: list[int]) -> torch.Tensor:

        # 0. Default height and width to unet
        if self.pipe._is_unet_config_sample_size_int:
            height = self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
            width = height
        else:
            height = self.pipe.unet.config.sample_size[0] * self.pipe.vae_scale_factor
            width = self.pipe.unet.config.sample_size[1] * self.pipe.vae_scale_factor

        # 2. Define call parameters
        batch_size = len(seeds)
        device = self.pipe._execution_device

        # 5. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        return self.pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.pipe.unet.dtype,
            device,
            [torch.Generator().manual_seed(seed) for seed in seeds],
            None,
        )

    def predict(self, x: torch.Tensor, t: torch.Tensor, cond: Condition) -> ModelOutput:

        latent_model_input = torch.cat([x, x])
        prompt_embeds = torch.cat(
            [cond["negative_prompt_embeds"], cond["prompt_embeds"]]
        )
        noise_pred_uncond, noise_pred_cond = self.pipe.unet(
            latent_model_input, t, encoder_hidden_states=prompt_embeds
        ).sample.chunk(2)
        noise_pred = noise_pred_uncond + cond["guidance_scale"] * (
            noise_pred_cond - noise_pred_uncond
        )

        pred_type = PredictionType(self.pipe.scheduler.config.prediction_type)
        return ModelOutput(pred=noise_pred, pred_type=pred_type)

    def postprocess(self, x: torch.Tensor) -> list[Image]:
        image = self.pipe.vae.decode(x / self.pipe.vae.config.scaling_factor).sample
        image = self.pipe.image_processor.postprocess(image)

        self.pipe.maybe_free_model_hooks()
        return image

    def save_output(self, output: Image, folder: str, index: int) -> None:
        save_png(output, folder, index)
