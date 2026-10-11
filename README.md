# RNR-GAD

## Environment

Python 3.10, PyTorch 2.8.0 (CUDA 12.9), Lightning 2.6.0, and PyG 2.7.0.
The verified CUDA extensions are `torch-scatter 2.1.2+pt28cu129` and
`torch-sparse 0.6.18+pt28cu129`.

Install PyTorch and the PyG extensions for your CUDA platform, then install the
remaining dependencies:

```sh
pip install -r requirements.txt
```

## Training

Run from the project root. Replace `xxx` with a configuration name from
`configs/experiment/`, without `.yaml`, such as `rnrgad_disney`.

Single-seed training:

```sh
python src/train.py experiment=xxx
```

Multi-seed training:

```sh
python src/multi_train.py experiment=xxx
```
