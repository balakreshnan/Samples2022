# Stable Diffusion in Azure Machine Learning

## Run Stable diffusion in Azure Machine Learning

## Prerequisites

- Azure subscription
- Azure Machine Learning Workspace
- Azure Storage Account
- GPU Compute cores

## Code

- Goal for this example is to run Stable diffusion in Azure Machine Learning
- Using example from - https://stability.ai/blog/stable-diffusion-public-release

```
!pip install diffusers==0.10.0
!pip install transformers scipy ftfy accelerate
!pip install "ipywidgets>=7,<8"
```

```
pip install diffusers==0.10.0
```

```
pip install transformers scipy ftfy accelerate
```

```
pip install "ipywidgets>=7,<8"
```

```
!python3 -m pip install --upgrade pip
```

```
!python3 -m pip install --upgrade Pillow
```

```
pip install Pillow
```

- now lets import diffusers

```
import torch
from diffusers import StableDiffusionPipeline

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)  
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/StableDiffusion/images/sd1.jpg "Architecture")

- Setup cuda to process inferencing

```
pipe = pipe.to("cuda")
```

- now let's call the stable diffusion model

```
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

# Now to display an image you can either save it such as:
image.save(f"astronaut_rides_horse.png")

# or if you're in a google colab you can directly display it with 
image.show()
```

