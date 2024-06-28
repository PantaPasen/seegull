$\color{#FF0000}\textsf{\Large\&#x26A0;\kern{0.2cm}\normalsize TODO: seegull logo}$ 

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

Using one of the pretrained models to detect an object and predict its object and material types:

$\color{#FF0000}\textsf{\Large\&#x26A0;\kern{0.2cm}\normalsize TODO: Add seegull models to huggingface and create a model downloader for this to work.}$ 

```
from seegull import Image, YOLOMultiLabel

model = YOLOMultiLabel("object-material-vX.pt")
image = Image(url="...")
image.predict(model).annotate().display()
```

![example annotation of a metal can](docs/example_annotation.png)

See the [notebooks](notebooks) folder for more thorough examples of how to use seegull.

## Models

$\color{#FF0000}\textsf{\Large\&#x26A0;\kern{0.2cm}\normalsize TODO: Short description of the released models with links where to find them.}$ 

## Data

$\color{#FF0000}\textsf{\Large\&#x26A0;\kern{0.2cm}\normalsize TODO: Short description of the released data with links where to find it and more info.}$ 
