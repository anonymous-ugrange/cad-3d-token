This repository contains the implementation of the methods described in **Bridging Tokens and Geometry: Token-wise 3D supervision for CAD Generation**.

## Environment
We recommend using a clean virtual environment.

To create the environment, run:

```bash
$ conda env create -f environment.yml
```
## Data Preparation
This section describes how to obtain and organize the datasets used in our experiments.

### Text2CAD
The Text2CAD dataset can be downloaded from [Text2CAD](https://github.com/SadilKhan/Text2CAD).

### Drawing2CAD
The Drawing2CAD dataset can be downloaded from [here](https://drive.google.com/file/d/1Sj8z1Sl8gkdHT0kNS3wabfWNyj7aNv7f/view?usp=drive_link).

## Training

### Text2CAD
In the `Cad_VLM/config/trainer_text2cad.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load BERT model weights from Huggingface.
- `cad_seq_dir`: The root directory that contains the ground truth CAD vector.
- `prompt_path`: Path for the text annotation.
- `split_filepath`: Json file containing the UIDs for train, test or validation.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path` (Optional): For resuming training after some epochs.

</p>
</details> 

<br>

```bash
$ cd Cad_VLM
$ CUDA_VISIBLE_DEVICES=<GPU_IDS> torchrun --nproc_per_node=<NUM_GPUS> train_text2cad.py --config_path config/trainer_text2cad.yaml
```

### Drawing2CAD
In the `Cad_VLM/config/trainer_drawing2cad.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cad_seq_dir`: The root directory that contains the ground truth CAD vector.
- `svg_dir`: Path for the SVG inputs.
- `split_filepath`: Json file containing the UIDs for train, test or validation.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path` (Optional): For resuming training after some epochs.

</p>
</details> 

<br>

```bash
$ cd Cad_VLM
$ CUDA_VISIBLE_DEVICES=<GPU_IDS> torchrun --nproc_per_node=<NUM_GPUS> train_drawing2cad.py --config_path config/trainer_drawing2cad.yaml
```

## Inference

### Text2CAD
In the `Cad_VLM/config/inference_text2cad.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load BERT model weights from Huggingface.
- `cad_seq_dir`: The root directory that contains the ground truth CAD vector.
- `prompt_path`: Path for the text annotation.
- `split_filepath`: Json file containing the UIDs for train, test or validation.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path`: The path to model weights. 

</p>
</details> 

<br>

```bash
$ cd Cad_VLM
$ CUDA_VISIBLE_DEVICES=<GPU_ID> python test_text2cad.py --config_path config/inference_text2cad.yaml
```
Results will be saved in `<log_dir>/test`

for evaluation, run:

```bash
$ cd Evaluation
$ python eval_seq.py --input_path <output_pkl> --output_dir <your_dir>
```

### Drawing2CAD
In the `Cad_VLM/config/inference_drawing2cad.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cad_seq_dir`: The root directory that contains the ground truth CAD vector.
- `svg_dir`: Path for the SVG inputs.
- `split_filepath`: Json file containing the UIDs for train, test or validation.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path` (Optional): For resuming training after some epochs.

</p>
</details> 

<br>

```bash
$ cd Cad_VLM
$ CUDA_VISIBLE_DEVICES=<GPU_ID> python test_drawing2cad.py --config_path config/inference_drawing2cad.yaml
```
Results will be saved in `<log_dir>/test`

for evaluation, run:

```bash
$ cd Evaluation
$ python eval_seq.py --input_path <output_pkl> --output_dir <your_dir>
```

## Acknowledgement
Our code is based on [Text2CAD](https://github.com/SadilKhan/Text2CAD) and [Drawing2CAD](https://github.com/lllssc/Drawing2CAD).