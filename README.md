# Depth_W
**Deploy the Depth Map on Web**

First install miniconda in your laptop


Installing on Mac
=====================
Refer to Windows.srt

Installing on Windows
=====================
Refer to Windows.srt

# Create new environment
## **Environment setup**
The project depends on :
- [pytorch](https://pytorch.org/) (Main framework)
- [timm](https://timm.fast.ai/)  (Backbone helper for MiDaS)
- pillow, matplotlib, scipy, h5py, opencv (utilities)

Install environment using `environment.yml` : 

Using conda : 

```bash
conda env create -n DemoDepth --file environment.yml
conda activate DemoDepth
```

## **Usage**
It is recommended to fetch the latest [MiDaS repo](https://github.com/isl-org/MiDaS) via torch hub before proceeding:
```python
import torch
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
```

### **ZoeDepth models** <!-- omit in toc -->
### Using torch hub
```python
import torch

repo = "isl-org/ZoeDepth"
# Zoe_N
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
```
### Using local copy
Clone this repo
#### Using local torch hub
You can use local source for torch hub to load the ZoeDepth models, for example: 
```python
import torch

# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
```

#### or load the models manually
```python
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# ZoeD_N
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)
```

### Using ZoeD models to predict depth 
```python
##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


# Local file
from PIL import Image
image = Image.open("/path/to/image.jpg").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

# Tensor 
from zoedepth.utils.misc import pil_to_batched_tensor
X = pil_to_batched_tensor(image).to(DEVICE)
depth_tensor = zoe.infer(X)

# From URL
from zoedepth.utils.misc import get_image_from_url

# Example URL
URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"
image = get_image_from_url(URL)  # fetch
depth = zoe.infer_pil(image)

# Save raw
from zoedepth.utils.misc import save_raw_16bit
fpath = "/path/to/output.png"
save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize
colored = colorize(depth)

# save colored output
fpath_colored = "/path/to/output_colored.png"
Image.fromarray(colored).save(fpath_colored)
```

## **Sanity checks** (Recommended)
Check if models can be loaded: 
```bash
python sanity_check.py
```
Try a demo prediction pipeline:
```bash
python sanity_check_begin.py
```
This will save a file `pred.png` in the root folder, showing RGB and corresponding predicted depth side-by-side.

## **Gradio demo**
To get started, install UI requirements:
```bash
pip install -r display_requirements.txt
```
Then launch the gradio UI:
```bash
python -m Demo.app
```
