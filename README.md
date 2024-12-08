# Enhancing Security of Proof-of-Learning Against Spoofing Attacks Using Advanced Model Watermarking Methods

**Abstract:**

The rapid advancement of machine learning (ML) technologies necessitates robust security frameworks to protect the integrity of ML model training processes. Proof-of-Learning (PoL) is a critical method for verifying the computational effort in training ML models, while model watermarking is a strategy for asserting model ownership. This research integrates PoL with advanced model watermarking techniques, including **Feature-Based**, **Parameter Perturbation-Based**, and **Non-Intrusive** watermarking methods. These integrations mitigate security risks associated with external key management and reduce computational overhead by eliminating the need for complex verification procedures. Our proposed dual-layered verification architecture embeds unique watermarks during the training phase. It records them alongside PoL proofs, enhancing security against sophisticated spoofing attacks where adversaries attempt to mimic a model's computational trajectory and watermark. This approach addresses key challenges, including maintaining watermark robustness and balancing security with model performance. Through a comprehensive analysis, we identify vulnerabilities in existing PoL systems and demonstrate how advanced watermarking can enhance security. We present a secure PoL mechanism, supported by empirical validation, that significantly improves resilience to spoofing attacks. This advancement represents a crucial step towards securing ML models, paving the way for future research to protect diverse ML applications from various threats.

**Keywords:**

Proof-of-Learning, Model Watermarking, Feature-Based Watermarking, Parameter Perturbation-Based Watermarking, Non-Intrusive Watermarking, Machine Learning Security, Spoofing Attack Countermeasures, Dual-Layered Verification, Model Authenticity, Intellectual Property Protection in ML, Computational Effort Authentication, Security Enhancements in Machine Learning, Watermark Robustness, Model Integrity Verification.

---

## Introduction

This repository builds upon the innovative work presented in ["Proof-of-Learning: Definitions and Practice"](https://arxiv.org/abs/2103.05633), featured at the 42nd IEEE Symposium on Security and Privacy. The paper introduces the Proof-of-Learning (PoL) concept in machine learning (ML), inspired by mechanisms of proof-of-work and verified computing. It details how the gradient descent algorithm—fundamental to ML training—accumulates stochastic information. It provides a basis for PoL to validate the computational work invested in accurately deriving a model's parameters.

Despite its strengths, the PoL framework has security vulnerabilities, particularly spoofing attacks that can significantly compromise the system's integrity. Our research enhances the PoL framework's resilience to such attacks by integrating advanced model watermarking techniques:

- **Feature-Based Watermarking**
- **Parameter Perturbation-Based Watermarking**
- **Non-Intrusive Watermarking**

This integration:

- **Mitigates security risks** associated with external key management.
- **Reduces computational overhead** by eliminating the need for complex verification procedures.
- **Enhances defense** against sophisticated spoofing attacks where adversaries attempt to mimic a model's computational trajectory and watermark.

We have thoroughly tested our code with datasets such as CIFAR-10, CIFAR-100, and a subset of ImageNet, ensuring the efficacy of our security enhancements. For an in-depth understanding of our contributions to fortifying the PoL framework against adversarial threats, we encourage you to review the aforementioned papers.

## Dependency

Our code is implemented and tested on PyTorch. The following packages are used:

```bash
torch==1.8.0
torchvision==0.9.0
numpy
scipy
```

### Training with Advanced Watermarking Techniques

To train a model, create a proof-of-learning, and embed a watermark using one of the supported watermarking methods, use the following command:

```bash
python PoL/train_with_watermark.py \
    --save-freq [checkpointing interval] \
    --dataset [dataset] \
    --model [model architecture] \
    --epochs [number of epochs] \
    --lambda-wm [watermark loss weight] \
    --k [watermark embedding frequency] \
    --watermark-key [secret key] \
    --watermark-method [watermarking method] \
    [additional parameters based on method]
```

- `--save-freq` is the checkpointing interval (denoted by *k* in the PoL paper).
- `--lambda-wm` controls the weight of the watermark loss relative to the main loss.
- `--k` determines how frequently the watermark is embedded during training steps.
- `--watermark-key` is a secret key used to generate the watermark.

Watermarking Methods
Feature-Based Watermarking (default)
Parameter Perturbation-Based Watermarking
Non-Intrusive Watermarking
Additional Parameters Based on Method
For Parameter Perturbation-Based Watermarking:
--num-parameters [number of parameters to perturb]
--perturbation-strength [strength of perturbations]
For Non-Intrusive Watermarking:
--watermark-size [size of the watermark]

**Example:**

Feature-Based Watermarking
```bash
python PoL/train_with_watermark.py \
    --save-freq 100 \
    --dataset CIFAR10 \
    --model resnet20 \
    --epochs 2 \
    --lambda-wm 0.01 \
    --k 1000 \
    --watermark-key 'secret_key' \
    --watermark-method 'feature_based'
```

Parameter Perturbation-Based Watermarking
```bash
python PoL/train_with_watermark.py \
    --save-freq 100 \
    --dataset CIFAR10 \
    --model resnet20 \
    --epochs 2 \
    --lambda-wm 0.01 \
    --k 1000 \
    --watermark-key 'secret_key' \
    --watermark-method 'parameter_perturbation' \
    --num-parameters 1000 \
    --perturbation-strength 1e-5
```

Non-Intrusive Watermarking
```bash
python PoL/train_with_watermark.py \
    --save-freq 100 \
    --dataset CIFAR10 \
    --model resnet20 \
    --epochs 2 \
    --lambda-wm 0.01 \
    --k 1000 \
    --watermark-key 'secret_key' \
    --watermark-method 'non_intrusive' \
    --watermark-size 128
```

### Verification

To verify a given proof-of-learning and check for the presence of the feature-based watermark:

```bash
python PoL/verify.py \
    --model-dir [path/to/the/proof] \
    --dataset [dataset] \
    --model [model architecture] \
    --epochs [number of epochs] \
    --save-freq [checkpointing interval] \
    --batch-size [batch size] \
    --lr [learning rate] \
    --lambda-wm [watermark loss weight] \
    --k [watermark embedding frequency] \
    --watermark-key [secret key] \
    --watermark-method [watermarking method] \
    [additional parameters based on method] \
    --dist [distance metrics] \
    --delta [thresholds] \
    --watermark-path [path to watermarked model]
```

- `--dist` can be one or more of `1`, `2`, `inf`, `cos` (separated by spaces).
- `--delta` are the corresponding thresholds for the distance metrics.
- `--watermark-path` specifies the path to the saved model with the embedded watermark.

**Examples:**
Feature-Based Watermarking

```bash
python PoL/train_with_watermark.py \
    --save-freq 100 \
    --dataset CIFAR10 \
    --model resnet20 \
    --epochs 2 \
    --lambda-wm 0.01 \
    --k 1000 \
    --watermark-key 'secret_key' \
    --watermark-method 'feature_based'
```

Parameter Perturbation-Based Watermarking

```bash
python PoL/train_with_watermark.py \
    --save-freq 100 \
    --dataset CIFAR10 \
    --model resnet20 \
    --epochs 2 \
    --lambda-wm 0.01 \
    --k 1000 \
    --watermark-key 'secret_key' \
    --watermark-method 'parameter_perturbation' \
    --num-parameters 1000 \
    --perturbation-strength 1e-5
```

Non-Intrusive Watermarking

```bash
python PoL/train_with_watermark.py \
    --save-freq 100 \
    --dataset CIFAR10 \
    --model resnet20 \
    --epochs 2 \
    --lambda-wm 0.01 \
    --k 1000 \
    --watermark-key 'secret_key' \
    --watermark-method 'non_intrusive' \
    --watermark-size 128
```

# Verification

To verify a given proof-of-learning and check for the presence of the watermark, use the following command:

```bash
python PoL/verify.py \
    --model-dir [path/to/the/proof] \
    --dataset [dataset] \
    --model [model architecture] \
    --epochs [number of epochs] \
    --save-freq [checkpointing interval] \
    --batch-size [batch size] \
    --lr [learning rate] \
    --lambda-wm [watermark loss weight] \
    --k [watermark embedding frequency] \
    --watermark-key [secret key] \
    --watermark-method [watermarking method] \
    [additional parameters based on method] \
    --dist [distance metrics] \
    --delta [thresholds] \
    --watermark-path [path to watermarked model]
```

Examples
Feature-Based Watermarking Verification
```bash
python PoL/verify.py \
    --model-dir proof/CIFAR10_Batch100 \
    --dataset CIFAR10 \
    --model resnet20 \
    --epochs 2 \
    --save-freq 100 \
    --batch-size 128 \
    --lr 0.1 \
    --lambda-wm 0.01 \
    --k 1000 \
    --watermark-key 'secret_key' \
    --watermark-method 'feature_based' \
    --dist 1 2 inf cos \
    --delta 10000 100 1 0.1 \
    --watermark-path model_with_feature_based_watermark.pth
```

Parameter Perturbation-Based Watermarking Verification
```bash
python PoL/verify.py \
    --model-dir proof/CIFAR10_Batch100 \
    --dataset CIFAR10 \
    --model resnet20 \
    --epochs 2 \
    --save-freq 100 \
    --batch-size 128 \
    --lr 0.1 \
    --lambda-wm 0.01 \
    --k 1000 \
    --watermark-key 'secret_key' \
    --watermark-method 'parameter_perturbation' \
    --num-parameters 1000 \
    --perturbation-strength 1e-5 \
    --dist 1 2 inf cos \
    --delta 10000 100 1 0.1 \
    --watermark-path model_with_parameter_perturbation_watermark.pth
```

Non-Intrusive Watermarking Verification
```bash
python PoL/verify.py \
    --model-dir proof/CIFAR10_Batch100 \
    --dataset CIFAR10 \
    --model resnet20 \
    --epochs 2 \
    --save-freq 100 \
    --batch-size 128 \
    --lr 0.1 \
    --lambda-wm 0.01 \
    --k 1000 \
    --watermark-key 'secret_key' \
    --watermark-method 'non_intrusive' \
    --watermark-size 128 \
    --dist 1 2 inf cos \
    --delta 10000 100 1 0.1 \
    --watermark-path model_with_non_intrusive_watermark.pth
```

### Run Sample

After integrating feature-based model watermarking into our training approach, we have observed significant improvements in both model performance and watermark robustness.

**Key Modifications:**

- **Integrated Training with Watermark Loss:** The watermark is embedded directly into the model's features during the normal training process by adding a watermark loss term to the primary loss function. This encourages the model to learn the watermark without degrading its performance on the primary task.

- **Feature-Based Watermarking:** Embedding the watermark in the model's internal representations enhances security against attacks that might strip away or modify external watermarks.

**Training Output:**

Below is the log information from a recent training run, demonstrating successful model training, watermark embedding, and Proof-of-Learning data generation.

<details>
<summary><strong>Click to expand the training log</strong></summary>

```bash
(venv) PS C:\dev\SecurePoL-Watermarking> python PoL/train_with_watermark.py --save-freq 100 --dataset CIFAR10 --model resnet20 --epochs 2 --lambda-wm 0.01 --k 1000 --watermark-key 'secret_key'
2024-09-26 13:04:17,120 - INFO - Trying to allocate 0 GPUs
2024-09-26 13:04:17,120 - INFO - Using device: cpu
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data\cifar-10-python.tar.gz
100.0%
Extracting ./data\cifar-10-python.tar.gz to ./data
2024-09-26 13:04:34,759 - INFO - Dataset loaded with 50000 samples.
2024-09-26 13:04:34,769 - INFO - Generated training sequence with length 100000
2024-09-26 13:04:34,859 - INFO - Data shape: (100000, 32, 32, 3), Data type: uint8
2024-09-26 13:04:34,860 - INFO - First data sample hash: a3f4c00fa8a122dbe09d61bc1b6f0649e0f0dd30f22239c25f1dc0cb2d9cdbb6
2024-09-26 13:04:37,107 - INFO - Computed hash during training: 40d8b2dd36df8ea21925e209d7305d262a7fdd7e670988f539080b2511019373
2024-09-26 13:04:37,107 - INFO - Saved dataset hash to hash.txt
2024-09-26 13:04:37,109 - INFO - Saved training sequence to indices.npy
2024-09-26 13:04:37,110 - INFO - Saved watermarking information to watermark_info.json
2024-09-26 13:04:37,110 - INFO - Model architecture: resnet20
2024-09-26 13:04:37,110 - INFO - Learning Rate: 0.1
2024-09-26 13:04:37,111 - INFO - Batch Size: 128
2024-09-26 13:04:37,111 - INFO - Epochs: 2
2024-09-26 13:04:37,111 - INFO - Optimizer: SGD
2024-09-26 13:04:37,111 - INFO - Scheduler: MultiStepLR with milestones [1, 1] and gamma 0.1
2024-09-26 13:04:37,117 - INFO - Saved initial model checkpoint at step 0
2024-09-26 13:04:37,118 - INFO - Starting epoch 1/2
2024-09-26 13:04:54,278 - INFO - Feature-based watermark loss computed at step 0
2024-09-26 13:05:37,232 - INFO - Saved checkpoint at step 100
2024-09-26 13:06:22,545 - INFO - Saved checkpoint at step 200
2024-09-26 13:07:05,266 - INFO - Saved checkpoint at step 300
2024-09-26 13:07:47,845 - INFO - Saved checkpoint at step 400
2024-09-26 13:08:30,903 - INFO - Saved checkpoint at step 500
2024-09-26 13:09:11,944 - INFO - Saved checkpoint at step 600
2024-09-26 13:09:53,800 - INFO - Saved checkpoint at step 700
2024-09-26 13:10:28,130 - INFO - Scheduler stepped at epoch 1/2
2024-09-26 13:10:28,130 - INFO - Starting epoch 2/2
2024-09-26 13:10:52,318 - INFO - Saved checkpoint at step 800
2024-09-26 13:11:32,919 - INFO - Saved checkpoint at step 900
2024-09-26 13:12:14,449 - INFO - Saved checkpoint at step 1000
2024-09-26 13:12:14,670 - INFO - Feature-based watermark loss computed at step 1000
2024-09-26 13:12:58,712 - INFO - Saved checkpoint at step 1100
2024-09-26 13:13:39,285 - INFO - Saved checkpoint at step 1200
2024-09-26 13:14:19,369 - INFO - Saved checkpoint at step 1300
2024-09-26 13:15:00,316 - INFO - Saved checkpoint at step 1400
2024-09-26 13:15:40,643 - INFO - Saved checkpoint at step 1500
2024-09-26 13:16:06,813 - INFO - Scheduler stepped at epoch 2/2
2024-09-26 13:16:06,826 - INFO - Saved final model checkpoint at step 1564
2024-09-26 13:16:06,853 - INFO - Starting feature-based watermark validation.
2024-09-26 13:16:07,009 - INFO - Feature-based watermark validation successful: Watermark detected.
2024-09-26 13:16:07,009 - INFO - Watermark Detection Accuracy: 100.00%
Files already downloaded and verified
2024-09-26 13:16:33,347 - INFO - Validation Accuracy: 74.90%
2024-09-26 13:16:33,359 - INFO - Model with watermark saved at model_with_watermark.pth
2024-09-26 13:16:33,359 - INFO - Total training time: 736.24 seconds
```

</details>

**Key Observations:**

- **Successful Training:** The model successfully trains over 2 epochs, saving checkpoints at specified intervals.
- **Watermark Embedding:** Feature-based watermark loss is computed at the specified steps (e.g., steps 0 and 1000), indicating that the watermark embedding is occurring as intended.
- **Watermark Validation:** The watermark validation at the end of training confirms that the watermark is successfully embedded with 100% detection accuracy.
- **Model Performance:** The final model achieves a validation accuracy of **74.90%**, demonstrating that the watermarking process does not significantly degrade model performance.

### Verification Output

Below is the log information from the verification run, demonstrating successful proof-of-learning verification and watermark detection.

<details>
<summary><strong>Click to expand the verification log</strong></summary>

```bash
(venv) PS C:\dev\SecurePoL-Watermarking> python PoL/verify.py --model-dir proof/CIFAR10_Batch100 --dataset CIFAR10 --model resnet20 --epochs 2 --save-freq 100 --batch-size 128 --lr 0.1 --lambda-wm 0.01 --k 1000 --watermark-key 'secret_key' --dist 1 2 inf cos --delta 10000 100 1 0.1 --watermark-path model_with_watermark.pth
2024-09-26 13:23:32,990 - INFO - Starting the verification process...
2024-09-26 13:23:32,990 - INFO - Verifying model initialization...
2024-09-26 13:23:33,100 - INFO - The proof-of-learning passed the initialization verification.
Files already downloaded and verified
2024-09-26 13:23:33,931 - INFO - Data shape: (100000, 32, 32, 3), Data type: uint8
2024-09-26 13:23:33,931 - INFO - First data sample hash: a3f4c00fa8a122dbe09d61bc1b6f0649e0f0dd30f22239c25f1dc0cb2d9cdbb6
2024-09-26 13:23:35,786 - INFO - Saved hash from training: 40d8b2dd36df8ea21925e209d7305d262a7fdd7e670988f539080b2511019373
2024-09-26 13:23:35,786 - INFO - Computed hash during verification: 40d8b2dd36df8ea21925e209d7305d262a7fdd7e670988f539080b2511019373
2024-09-26 13:23:35,786 - INFO - Hash of the proof is valid.
2024-09-26 13:23:35,786 - INFO - Performing top-q verification...
2024-09-26 13:23:35,802 - INFO - Verifying epoch 1/2
2024-09-26 13:23:51,282 - INFO - Feature-based watermark loss computed at step 0
2024-09-26 13:24:30,811 - INFO - Scheduler stepped at epoch 1/1
2024-09-26 13:25:27,584 - INFO - Scheduler stepped at epoch 1/1
2024-09-26 13:26:25,214 - INFO - Distance metric: 1 || threshold: 10000.0 || Q=2
2024-09-26 13:26:25,214 - INFO - Average top-q distance: 0.0
2024-09-26 13:26:25,214 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-09-26 13:26:25,214 - INFO - Distance metric: 2 || threshold: 100.0 || Q=2
2024-09-26 13:26:25,214 - INFO - Average top-q distance: 0.0
2024-09-26 13:26:25,214 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-09-26 13:26:25,214 - INFO - Distance metric: inf || threshold: 1.0 || Q=2
2024-09-26 13:26:25,214 - INFO - Average top-q distance: 0.0
2024-09-26 13:26:25,214 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-09-26 13:26:25,214 - INFO - Distance metric: cos || threshold: 0.1 || Q=2
2024-09-26 13:26:25,214 - INFO - Average top-q distance: -4.490216573079427e-06
2024-09-26 13:26:25,214 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-09-26 13:26:25,214 - INFO - Verifying epoch 2/2
2024-09-26 13:26:42,705 - INFO - Scheduler stepped at epoch 1/1
2024-09-26 13:26:59,722 - INFO - Distance metric: 1 || threshold: 10000.0 || Q=2
2024-09-26 13:26:59,722 - INFO - Average top-q distance: 78.73853874206543
2024-09-26 13:26:59,722 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-09-26 13:26:59,722 - INFO - Distance metric: 2 || threshold: 100.0 || Q=2
2024-09-26 13:26:59,722 - INFO - Average top-q distance: 0.241085484623909
2024-09-26 13:26:59,722 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-09-26 13:26:59,722 - INFO - Distance metric: inf || threshold: 1.0 || Q=2
2024-09-26 13:26:59,722 - INFO - Average top-q distance: 0.013262152671813965
2024-09-26 13:26:59,722 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-09-26 13:26:59,722 - INFO - Distance metric: cos || threshold: 0.1 || Q=2
2024-09-26 13:26:59,722 - INFO - Average top-q distance: 8.13603401184082e-06
2024-09-26 13:26:59,722 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-09-26 13:26:59,722 - INFO - Verifying watermark presence in the model...
2024-09-26 13:26:59,753 - INFO - Starting feature-based watermark validation.
2024-09-26 13:26:59,926 - INFO - Feature-based watermark validation successful: Watermark detected.
2024-09-26 13:26:59,926 - INFO - Feature-based watermark verification successful: Watermark is present in the model.
2024-09-26 13:26:59,926 - INFO - Verification process concluded successfully.
```

</details>

**Key Observations:**

- **Initialization Verification:** The proof-of-learning passes the initialization verification and hash verification, confirming the integrity of the initial model and dataset.
- **Top-Q Verification:** During top-q verification for both epochs, none of the steps exceed the specified thresholds for any of the distance metrics, indicating that the proof-of-learning is valid.
- **Watermark Verification:** The feature-based watermark validation is successful, with the watermark detected in the model.
- **Conclusion:** The verification process concludes successfully, affirming the model's authenticity and the validity of the computational effort.

### Conclusion

The integration of feature-based model watermarking into the PoL framework has been successful. The model maintains high performance on its primary task while effectively embedding and verifying the watermark. This enhancement strengthens the security of PoL against spoofing attacks and unauthorized model use.

### Spoofing Attacks

To simulate spoofing attacks and test the robustness of the PoL framework with feature-based watermarking, you can use the following commands:

**For CIFAR-10 and CIFAR-100:**

```bash
python spoof_cifar/attack.py --attack [1,2, or 3] --dataset [CIFAR10 or CIFAR100] --model [model architecture] --t [spoof steps] --verify [1 or 0]
```

- `--attack` selects the type of attack.
- `--t` specifies the number of spoof steps (denoted by *T* in the paper).
- `--verify` controls whether to verify the model after the attack.

**For ImageNet Subset:**

```bash
python spoof_imagenet/spoof_imagenet.py --t [spoof steps] --verify [1 or 0]
python spoof_imagenet/spoof_attack3_imagenet.py --t [spoof steps] --verify [1 or 0]
```

### Model Generation

To generate models with high accuracy for spoofing experiments:

```bash
python spoof_cifar/train.py --save-freq [checkpointing interval] --dataset [CIFAR10 or CIFAR100] --model [resnet20 or resnet50]
```

### Verification of Spoofed Models

To verify a given proof-of-learning or a given spoofed model:

```bash
python PoL/verify.py --model-dir [path/to/the/proof] --dist [distance metrics] --q [query budget] --delta [thresholds]
```

Ensure that the parameters `lr`, `batch-size`, `epochs`, `dataset`, `model`, and `save-freq` are consistent with those used during training.

---

For more detailed information about the training and verification processes, refer to the code and comments in the repository. If you encounter any issues or have questions, please open an issue, and we'll be happy to assist.
