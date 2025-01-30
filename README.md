<center>
<h2>HyLiFormer: Hyperbolic Linear Attention for Skeleton-based Human Action Recognition</h2>
</center>


---

<h5 align="left">
This repository is the official PyTorch implementation of "HyLiFormer: Hyperbolic Linear Attention for Skeleton-based Human Action Recognition". 
</h5>



## Requirements

> - Python >= 3.8.19
> - PyTorch >= 1.11.0
> - Platforms: Ubuntu 20.04, CUDA 11.3
> - We have included a dependency file for our experimental environment. To install all dependencies, create a new Anaconda virtual environment and execute the provided file. Run `conda env create -f requirements.yaml`.
> - Run `pip install -e torchlight`.

## Data Preparation

### Download datasets

#### There are 3 datasets to download:

- NTU RGB+D
- NTU RGB+D 120

#### NTU RGB+D and NTU RGB+D 120

1. Request dataset from [here](https://rose1.ntu.edu.sg/dataset/actionRecognition)
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

### Data Processing

#### Directory Structure

- Put downloaded data into the following directory structure:

```
- data/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```


## Training
```bash
# Train HyLiFormer on NTU RGB+D X-Sub60 dataset (joint modality)
python main.py --config ./config/train/ntu_cs/HyLiFormer_j.yaml

# Train HyLiFormer on NTU RGB+D X-Set60 dataset (joint modality)
python main.py --config ./config/train/ntu_cv/HyLiFormer_j.yaml

# Train HyLiFormer on NTU RGB+D X-Sub120 dataset (joint modality)
python main.py --config ./config/train/ntu120_csub/HyLiFormer_j.yaml

# Train HyLiFormer on NTU RGB+D X-Set120 dataset (joint modality)
python main.py --config ./config/train/ntu120_cset/HyLiFormer_j.yaml 
```

## Testing
```bash
# Test HyLiFormer on NTU RGB+D X-Sub60 dataset (joint modality)
python main.py --config ./config/test/ntu_cs/HyLiFormer_j.yaml

# Test HyLiFormer on NTU RGB+D X-Set60 dataset (joint modality)
python main.py --config ./config/test/ntu_cv/HyLiFormer_j.yaml

# Test HyLiFormer on NTU RGB+D X-Sub120 dataset (joint modality)
python main.py --config ./config/test/ntu120_csub/HyLiFormer_j.yaml

# Test HyLiFormer on NTU RGB+D X-Set120 dataset (joint modality)
python main.py --config ./config/test/ntu120_cset/HyLiFormer_j.yaml 

```
