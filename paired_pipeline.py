"""Native EditPlus image conditioning with persistent target-only ERP padding."""
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from .paired_circular import install_circular_padding
from .circular import pad_columns, crop_columns


class PairedPanoPipeline(QwenImageEditPlusPipeline):
    padding_columns = 0

    def enable_pano(self, columns):
        install_circular_padding(self.transformer, columns, mode='persistent')
        self.padding_columns = columns

    def _encode_vae_image(self, image, generator):
        # Keep reference VAE fp32 as in cache; return bf16 for transformer concat.
        return super()._encode_vae_image(image.to(self.vae.dtype), generator).to(image.dtype)

    def prepare_latents(self, images, batch_size, num_channels_latents, height, width,
                        dtype, device, generator, latents=None):
        target, reference = super().prepare_latents(images, batch_size, num_channels_latents,
                         height, width, dtype, device, generator, latents)
        if self.padding_columns:
            target = pad_columns(target,height//16,width//16,self.padding_columns)
        return target, reference

    @torch.no_grad()
    def __call__(self, image=None, prompt=None, *, height=1024, width=2048,
                 output_type='pil', return_dict=True, **kwargs):
        if image is None or height <= 0 or height % 32 or width != 2*height:
            raise ValueError('Supply Local RGB and a 2:1 ERP size, height divisible by 32')
        result = super().__call__(image=image,prompt=prompt,height=height,width=width,
                                  output_type='latent',return_dict=False,**kwargs)[0]
        if self.padding_columns:
            result = crop_columns(result,height//16,width//16,self.padding_columns)
        if output_type == 'latent':
            images = result
        else:
            z = self._unpack_latents(result,height,width,self.vae_scale_factor).to(self.vae.dtype)
            mean = torch.tensor(self.vae.config.latents_mean,device=z.device,dtype=z.dtype).view(1,-1,1,1,1)
            std = torch.tensor(self.vae.config.latents_std,device=z.device,dtype=z.dtype).view(1,-1,1,1,1)
            decoded = self.vae.decode(z*std+mean,return_dict=False)[0][:,:,0]
            images = self.image_processor.postprocess(decoded,output_type=output_type)
        self.maybe_free_model_hooks()
        return QwenImagePipelineOutput(images=images) if return_dict else (images,)
