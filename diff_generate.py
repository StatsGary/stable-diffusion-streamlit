import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

class StableDiffusionLoader:
    def __init__(self, 
                prompt:str, 
                pretrain_pipe:str='CompVis/stable-diffusion-v1-4'):
        self.prompt = prompt
        self.pretrain_pipe = pretrain_pipe
        assert isinstance(self.prompt, str), 'Please enter a string into the prompt field'
        assert isinstance(self.pretrain_pipe, str), 'Please use value such as `CompVis/stable-diffusion-v1-4` for pretrained pipeline'


    def generate_image_from_prompt(self, save_location='prompt.jpg', use_token=False,
                                   verbose=False):
        pipe = StableDiffusionPipeline.from_pretrained(self.pretrain_pipe, 
            revision="fp16", torch_dtype=torch.float16, 
            use_auth_token=use_token)

        pipe = pipe.to('cuda')
        with autocast('cuda'):
            image = pipe(self.prompt)[0][0]
        if verbose: 
            print('[INFO] saving image to desired location')

        image.save(save_location)
        return image    

    def __str__(self) -> str:
        return f'Generating image for prompt {self.prompt}'

if __name__ == '__main__':
    # Intantiate class with relevant prompt
    st = StableDiffusionLoader('homer simpson on the computer keyboard wearing a space suit')
    st.generate_image_from_prompt(verbose=True)

