import pytorch_lightning as pl

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from unet.unet import UNet2DConditionModel
from typing import List, Optional, Union
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

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
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(
            scheduler.set_timesteps).parameters.keys())
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


class SDXLArchitectModule(pl.LightningModule):
    def __init__(self, args, ):
        super().__init__()
        # model_path = "checkpoints/v1-5-pruned.safetensors"
        model_name = "stabilityai/stable-diffusion-xl-base-1.0"

        # self.pipe = StableDiffusionPipeline.from_single_file(
        #     model_path,
        # ).to('cuda')

        self.args = args

        height = None
        width = None

        self.do_classifier_free_guidance = True

        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            model_name, subfolder='unet'
        )

        # self.unet.load_state_dict(self.unet_state_dict)
        # self.controlnet = self.unet.clone()

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_name, subfolder='vae'
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.height = height or self.unet.config.sample_size * self.vae_scale_factor
        self.width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.noise_scheduler =  DDPMScheduler.from_pretrained(
            model_name, subfolder="scheduler", 
            rescale_betas_zero_snr=True)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer_2")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
           model_name, subfolder="text_encoder_2")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

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
        pixel_values = batch["image"]
        prompt = batch["prompt"]
        negative_prompt = batch["negative_prompt"]

        model_input = self.vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * self.vae.config.scaling_factor

        noise = torch.randn_like(model_input)

        bsz = model_input.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)

        noisy_latents = self.noise_scheduler.add_noise(
            model_input, noise, timesteps)

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
        )

        # This is for classifier free guidance. We are treating negative prompt as unconditional tokens if negative prompt is provided.
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        noise_pred = self.forward(noisy_latents, timesteps, prompt_embeds)

        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        return loss
    
    def _get_add_time_ids(
        self, original_size, 
        crops_coords_top_left, 
        target_size, dtype, 
        text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

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
                 negative_prompt="low res",
                 prompt_2 = None, 
                 negative_prompt_2 = None,
                 control_net_input=None,
                 num_inference_steps=50,
                 num_images_per_prompt=1,
                 guidance_scale=6.0):
        with torch.no_grad():
            self.check_inputs(prompt,
                              negative_prompt=negative_prompt
                              )
            self.guidance_scale = guidance_scale

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)

            (prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
            )

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

            add_text_embeds = pooled_prompt_embeds
            if self.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
            
            add_time_ids = self._get_add_time_ids(
                (self.height, self.width),
                (0, 0),
                (self.height, self.width),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )

            negative_add_time_ids = add_time_ids

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(self.device)
            add_text_embeds = add_text_embeds.to(self.device)
            add_time_ids = add_time_ids.to(self.device).repeat(batch_size * num_images_per_prompt, 1)

            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat(
                    [latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.noise_scheduler.scale_model_input(
                    latent_model_input, t)

                # predict the noise residual
                # noise_pred = self.unet(
                #     latent_model_input,
                #     t,
                #     encoder_hidden_states=prompt_embeds,
                #     return_dict=False,
                # )[0]
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.forward(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                )

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False,)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            # temp squuze. fix later
            image = (image.squeeze(0).detach().cpu().permute(
                1, 2, 0).numpy() * 255).astype("uint8")

            return Image.fromarray(image)

    def forward(self, latent, timestep, encoder_hidden_states, added_cond_kwargs = None):
        return self.unet(
            latent,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )[0]

    def encode_prompt(
        self,
        prompt,
        prompt_2 = None,
        negative_prompt=None,
        negative_prompt_2 = None,
        num_images_per_prompt=1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        prompt_embeds_list = []
        prompts = [prompt, prompt_2]

        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
           
            prompt_embeds = text_encoder(text_input_ids.to(self.device), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            
            prompt_embeds = prompt_embeds.hidden_states[-2]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        zero_out_negative_prompt = negative_prompt is None
        if self.do_classifier_free_guidance and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif self.do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]


            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(self.device),
                    output_hidden_states=True,
                )
               
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

            if self.text_encoder_2 is not None:
                prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=self.device)
            else:
                prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=self.device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            if self.do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                if self.text_encoder_2 is not None:
                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=self.device)
                else:
                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=self.device)

                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        if self.do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
            

    def check_inputs(
        self,
        prompt,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
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
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

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
