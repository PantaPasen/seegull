# seegull

seegull is the library that powers [Bower's](https://getbower.com/) barcodeless scanning functionality.
It consists of code for training object detection models as well as trained models and the data.

## Installation

Install the prerequisites:
- `libgl1`
- `libusb-1.0-0-dev`

$\color{#FF0000}\textsf{\Large\&#x26A0;\kern{0.2cm}\normalsize TODO: Add seegull to pypi for this to work}$ 

Then nstall with `pip install seegull`.

## Usage

Using one of the pretrained models:

$\color{#FF0000}\textsf{\Large\&#x26A0;\kern{0.2cm}\normalsize TODO: Add seegull models to huggingface and create a model downloader for this to work.}$ 

```
from seegull import Image, YOLOMultiLabel

model = YOLOMultiLabel("object-material-vX.pt")
image = Image(url="...")
image.predict(model).annotate().display()
```

See the [notebooks](notebooks) folder for more thorough examples of how to use seegull.
