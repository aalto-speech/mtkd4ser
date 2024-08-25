<h1 align="center">Multi-Teacher Language-Aware Knowledge Distillation for Speech Emotion Recognition</h1>

<p align="center">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging_Face-blue" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/%F0%9F%94%A5-PyTorch-purple" alt="PyTorch" />
</p>

<div align="center">
  <details>
    <summary>Table of Contents</summary>
    <a href="#installation">Installation</a><br>
    <a href="#usage">Usage</a><br>
    <a href="#contributing">Contributing</a><br>
    <a href="#citation">Citation</a>
  </details>
</div>

## Installation

**1. Clone the Repository**
```bash
git clone https://github.com/aalto-speech/mtkd4ser.git
```
```bash
cd mtkd4ser
```

**2. Create the Environment**
```bash
conda env create -f environment.yml
```

**3. Activate the Environment**
```bash
conda activate ser_venv
```

<!--
**x. xx**
```bash

```
-->

## Usage


<!-- Available Configurations and Choices -->
It supports a range of configurable parameters for training, validation, and evaluation. The table below details each *Configuration* and its *options*. Select the options that fit your use case.
| **Configuration** | **Options**                       |
|:------------------|:----------------------------------|
| LINGUALITY        | `Monolingual` or `Multilingual`   |
| LANGUAGE          | `EN` or `FI` or `FR`              |
| PARADIGM          | `MTKD` or `KD` or `FT`            |
| TRAINING          | `1` or `0`                        |
| SESSION           | `EN: 1-5` or `FI: 1-9` or `FR: 1` |
| N_EPOCHS          | `ℤ⁺`                              |
| BATCH_SIZE        | `ℤ⁺`                              |
| LEARNING_RATE     | `ℝ⁺`                              |


## Contributing


## Citation



