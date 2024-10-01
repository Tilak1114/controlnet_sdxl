import pytorch_lightning as pl
from diffusers import StableDiffusionPipeline
import torch
from diffusers.models import UNet2DConditionModel
from typing import List, Optional, Union
import torch.nn.functional as F
from PIL import Image

import inspect

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class DiffusionArchitectModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        model_path = "checkpoints/v1-5-pruned.safetensors"
        
        self.pipe = StableDiffusionPipeline.from_single_file(
            model_path,
        ).to('cuda')

        self.args = args

        self.do_classifier_free_guidance = True

        self.unet: UNet2DConditionModel = self.pipe.unet
        self.vae = self.pipe.vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.height = self.unet.config.sample_size * self.vae_scale_factor
        self.width = self.unet.config.sample_size * self.vae_scale_factor

        self.noise_scheduler = self.pipe.scheduler

        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_class = torch.optim.AdamW

        return optimizer_class(
            self.parameters(),
            lr=self.lr,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
    
    def training_step(self, batch):
        # pixel_values = batch["image"]
        # prompt = batch["prompt"]
        # negative_prompt = batch["negative_prompt"]

        # model_input = self.vae.encode(pixel_values).latent_dist.sample()
        # model_input = model_input * self.vae.config.scaling_factor

        # noise = torch.randn_like(model_input)

        # bsz = model_input.shape[0]
        # timesteps = torch.randint(
        #     0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)

        # noisy_latents = self.noise_scheduler.add_noise(
        #     model_input, noise, timesteps)
        
        # prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        #     prompt,
        #     negative_prompt,
        # )

        # # This is for classifier free guidance. We are treating negative prompt as unconditional tokens if negative prompt is provided.
        # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # noise_pred = self.forward(noisy_latents, timesteps, prompt_embeds)

        # loss = F.mse_loss(noise_pred, noise, reduction="mean")
        pass
        

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if latents is None:
            latents = torch.randn(shape, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents
    
    def generate(self, prompt, 
                negative_prompt = "low res", 
                control_net_input = None,
                num_inference_steps=50,
                guidance_scale=6.0):
        with torch.no_grad():
            self.check_inputs(prompt,
                            self.height,
                            self.width, 
                            negative_prompt=negative_prompt
                            )
            self.guidance_scale = guidance_scale

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt,
            )

            # This is for classifier free guidance. We are treating negative prompt as unconditional tokens if negative prompt is provided.
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            timesteps, num_inference_steps = retrieve_timesteps(
                self.noise_scheduler, num_inference_steps, self.device, None, None
            )

            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * 1,
                num_channels_latents,
                self.height,
                self.width,
                prompt_embeds.dtype,
                self.device,
            )

            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                # noise_pred = self.unet(
                #     latent_model_input,
                #     t,
                #     encoder_hidden_states=prompt_embeds,
                #     return_dict=False,
                # )[0]

                noise_pred = self.forward(
                    latent_model_input, 
                    t,
                    prompt_embeds,
                    )

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False,)[0]
            image = (image/ 2 + 0.5).clamp(0, 1)
            #temp squuze. fix later
            image = (image.squeeze(0).detach().cpu().permute(1,2,0).numpy() * 255).astype("uint8")

            return Image.fromarray(image)

    
    def forward(self, latent, timestep, encoder_hidden, hint_img):
        return self.unet(
            latent,
            timestep,
            encoder_hidden,
            return_dict=False
        )[0]
    
    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        attention_mask = None

        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), attention_mask=attention_mask)
        prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * 1, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            