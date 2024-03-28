# Installation Instructions
- Clone this repository: `git clone https://github.com/r-karimi/vesicle-picker.git`
- Enter this repository: `cd vesicle_picker`
- Create a clean python virtual environment with venv or conda, tested with Python version 3.9 but likely compatible with Python 3.X.
	- With venv: `python3 -m venv .` and `source venv/bin/activate`.
	- With conda: `conda create -n vesicle-picker`, `conda activate vesicle-picker`, `conda install python=3.9`, and `conda install pip`.	
- Edit the `pyproject.toml` file to install the correction version of PyTorch, PyTorch vision, and PyTorch audio for your hardware, matching both your version of CUDA (e.g. `cu118` for CUDA 11.8) and Python version (e.g. `cp39` for Python 3.9). Available PyTorch wheels are listed [here](https://download.pytorch.org/whl/torch/). GPU acceleration is highly recommended when possible.
- Install `vesicle-picker` package and dependencies with:
	- `pip install .`
	- `poe install-pytorch`
- Place segment-anything model weights in repo base directory. They can be downloaded [here](https://github.com/facebookresearch/segment-anything#model-checkpoints).
- Modify csparc_login.ini to match your active CryoSPARC instance

Do next:
- Figure out how to subtract vesicles by extraction of a portion of the image from the mask bounding boxes to be more memory efficient (DONE)
- Add the non-vesicle background back into the subtracted vesicles to try to reduce subtraction artifacts (DONE)
- Dilate the masks by a few pixels before subtraction
- Write a subtracted micrograph export function that creates a cryosparc dataset where the micrograph paths point to the subtracted micrographs in an output folder

Minor fixes:
- Figure out whether model can be run on CPU in `initialize_model()`.

Longer-term:
- Make three scripts runable from the command line to: 
	1) take a cryosparc login info, project id, job id, job type, downsample factor, lowpass mode, and SAM kwargs to generate segmented micrographs, plot them, and save the masks and thumbnail images to an output folder (alongside originals and blurred images?).
	2) take an input directory of masks and thumbnail images (output of 1 above) and apply mask postprocessing, filtering for roundness, area, etc. Eventually, also compute vesicle statistics after postprocessing, like # detected per micrograph and distribution of roundnessess and areas and make nice plots in an output folder. Output folder should look like output/postprocessed_masks and output/thumbnails
	3) take a cryosparc login info, project id, job id, job type, take in input directory of postprocessed_masks, picker mode and box size to generate evenly spaced vesicle picks and dump each dataset into an output folder.
- All of these scripts should take inputs in Angstrom scale whenever necessary.
