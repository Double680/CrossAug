# CrossAug

This is the official repository of the SIGIR'24 paper "Cross-reconstructed Augmentation for Dual-target Cross-domain Recommendation".

## Prerequisites

Environments: python==3.9.18, pytorch==2.1.2, cuda==12.2

Install python packages via:
```
pip install -r requirements.txt
```

## Datasets

### Downloading

From [Amazon Review Data (2018)](https://jmcauley.ucsd.edu/data/amazon_v2/index.html), download "metadata" and "ratings only" for:
- Movie (Movies and TV) into `datasets/raw/Amazon/Movie`
- Music (CDs and Vinyi) into `datasets/raw/Amazon/Music`
- Cell (Cell Phones and Accessories) into `datasets/raw/Amazon/Cell`
- Elec (Electronics) into `datasets/raw/Amazon/Elec`

### Filtering

Entering `datasets` folders of this repo.
```{bash}
cd datasets
```

Then execute filtering command (with setting `--domain` to specify dataset):
```{bash}
python filter.py
```
The script takes inputs of filtered files from `raw` folder and save outputs into `processed` folder.

### Processing

After filtering, execute processing command (with setting `--domains` to specify dual datasets):
```{bash}
python process.py
```
The script takes inputs of filtered files from `filtered` folder and save outputs into `processed` folder.

## Implementation Details

Please refer to `config.yaml` and check the model configuration and hyperparameters.

## Running

Go back to the main directory and the execute the `main.py` to train & valid & test the model (with setting `--domains` to specify dual datasets):
```{bash}
python main.py
```

Result logs can be visualized with `--wandb`.

## Citation

Please cite our paper in the following format if you use our code during your research.

```
@inproceedings{2024crossaug,
  title={Cross-reconstructed Augmentation for Dual-target Cross-domain Recommendation},
  author={Qingyang Mao, Qi Liu, Zhi Li, Likang Wu, Bing Lv and Zheng Zhang},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  pages={2352--2356},
  year={2024}
}
```
