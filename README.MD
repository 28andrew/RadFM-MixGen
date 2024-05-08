# Multi-Modal Data Augmentation for Radiology Report Generation

This is the code for our approach, including dataset creation, model fine-tuning, and model inferencing, to the Shared task on Large-Scale Radiology Report Generation @ BioNLP ACL'24. The code is adapted from [RadFM](https://github.com/chaoyi-wu/RadFM/). We apply data augmentation by mixing images and text between pairs within the original dataset.

# Environment Setup
- Create a Conda environment based on the provided `environment.yml`.
- Install [en_score_sci_lg==0.5.3](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz) via pip.

# Dataset Download and MixGen Generation
- Complete the required training to gain access to the competition dataset, [interpret-cxr-public](https://huggingface.co/datasets/StanfordAIMI/interpret-cxr-public).
- Obtain an access token from your HuggingFace account.
- Within the `dataset` folder, add this token to `save_partial.py` and execute this code to save the dataset to disk.
- Use the notebooks `mix0.ipynb` to `mix3.ipynb` with their instructions to generate the synthetic mixed dataset. Note that this involves paid OpenAI API calls.

# RadFM Checkpoint Download and Conversion
- Download the [RadFM checkpoint](https://huggingface.co/chaoyi-wu/RadFM/tree/main) and save to `model/src/pytorch_model.bin`
- Run `model/src/save.py` to save this checkpoint in HuggingFace format to `model/src/Base/`

# Training
- Change the working directory to `model/src`
- Run `finetune_mixed.py`. It will automatically use all visible GPUs and resume from saved checkpoints as necessary. Checkpoints will be saved to `model/output`.

# Testing
- Change the working directory to `model/src`
- Run `inference_mixed.py` with the checkpoint number as the only argument and the inferences will be saved to `model/output` in files where their names indicate the checkpoint and GPU used for inference. The inferences are saved in multiple files, with one per GPU used for inference.
- Run `eval_mixed.py` with the checkpoint number as the only argument to calculate evaluation metrics on the inference outputs.