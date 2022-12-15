import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

class StableDiffusionLoader:
    def __init__(self, prompt, pretrain_pipe='CompVis/stable-diffusion-v1-4'):
        self.prompt = prompt
        self.pretrain_pipe = pretrain_pipe

    def create_pipe(self):
        pipe = StableDiffusionPipeline.from_pretrained(self.pretrain_pipe, 
            revision="fp16", torch_dtype=torch.float16, 
            use_auth_token=False)

        pipe = pipe.to('cuda')
        return pipe

    def generate_image_from_prompt(self, prompt, pipe):
        with autocast('cuda'):
            image = pipe(prompt)[0][0]
        image.save(f'prompt.jpg')
        return image    


st = StableDiffusionLoader('man on crutches')
st.create_pipe()
st.generate_image_from_prompt()



# PROMPT = 'a man on a scooter'

# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
#     revision="fp16", torch_dtype=torch.float16, 
#     use_auth_token=False)

# pipe = pipe.to('cuda')

# prompt = PROMPT
# with autocast('cuda'):
#     image = pipe(prompt)[0][0]

# image.save(f'prompt.jpg')



