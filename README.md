# Enhancing Security of Proof-of-Learning Against Spoofing Attacks Using Model Watermarking

**Abstract:**  
The rapid advancement of machine learning (ML) technologies has highlighted the need for robust security frameworks to protect the integrity of ML model training processes. Proof-of-Learning (PoL), which verifies the computational effort in training ML models, and model watermarking, a strategy for asserting model ownership, are critical. This research integrates PoL with model watermarking to enhance security against sophisticated spoofing attacks and unauthorized use. We propose a dual-layered verification architecture that embeds unique watermarks during training and documents these alongside PoL proofs. This approach authenticates computational effort and ensures model authenticity through watermark detection, strengthening defenses against spoofing. Spoofing attacks, where adversaries mimic a model's computational trajectory and watermark, pose a significant threat. We address challenges such as maintaining watermark robustness and balancing security with model performance. Our analysis reveals vulnerabilities in existing PoL systems and demonstrates how tailored watermarking strategies can fortify security. We present a provably secure PoL mechanism, supported by empirical validations, that enhances resilience to spoofing attacks. This advancement marks a significant step towards verifying the security of ML models, paving the way for further research to protect diverse ML applications from various threats.

## Enhancements in Proof-of-Learning with Adversarial Examples

This repository is anchored in the innovative work presented in ["Proof-of-Learning: Definitions and Practice"](https://arxiv.org/abs/2103.05633), featured at the 42nd IEEE Symposium on Security and Privacy. The paper introduces the Proof-of-Learning (PoL) concept in machine learning (ML), inspired by the mechanisms of proof-of-work and verified computing. It details how the gradient descent algorithm—fundamental to ML training—accumulates stochastic information. It provides a basis for PoL to validate the computational work invested in accurately deriving a model's parameters.

Despite its strengths, the PoL framework has security vulnerabilities, particularly spoofing attacks that can significantly compromise the system's integrity. Our research is committed to enhancing the PoL framework's resilience to such attacks.

In addition to the original PoL framework, this repository also integrates findings from the study on ["Adversarial Examples for Proof-of-Learning"](https://arxiv.org/abs/2108.09454). That paper presents a method that successfully uncovers PoL vulnerabilities by leveraging adversarial examples, thereby questioning the PoL's security posture. This vital research informs our efforts to bolster the PoL framework, addressing and mitigating these emergent threats.

Building on this, our repository further includes the implementation from ["Adversarial Examples for Proof-of-Learning"](https://arxiv.org/abs/2108.09454), which introduces a method that effectively attacks the PoL concept through adversarial examples. This approach challenges the security assumptions of the initial PoL proposition and guides our initiatives to improve PoL's robustness.

We have thoroughly tested our code with datasets such as CIFAR-10, CIFAR-100, and a subset of ImageNet, ensuring the efficacy of our security enhancements. For an in-depth understanding of our contributions to fortifying the PoL framework against adversarial threats, we would like you to review the abovementioned papers.

### Dependency
Our code is implemented and tested on PyTorch. The following packages are used:

```
torch==1.8.0 torchvision==0.9.0 numpy scipy
```
### Train
To train a model and create a proof-of-learning:
```
python train.py --save-freq [checkpointing interval] --dataset [any dataset in torchvision] --model [models defined in model.py or any torchvision model]
```
`save-freq` is a checkpointing interval, denoted by k in the paper. You can find a few other arguments at the end of the script.

Note that the proposed algorithm does not interact with the training process so that it could be applied to any kinds of gradient-descent based models.

### Verify
To verify a given proof-of-learning:
```
python verify.py --model-dir [path/to/the/proof] --dist [distance metric] --q [query budget] --delta [slack parameter]
```
Setting q to 0 or smaller will verify the whole proof; otherwise, the top-q iterations for each epoch will be verified. More information about q and delta can be found in the paper. For dist, you could use one or more of 1, 2, inf, cos (if more than one, separate them by space). The first 3 correspond to \(L_p\) norms, while cos is the cosine distance. Note that if using more than one, the top-q iterations for all distance metrics will be verified.

Please ensure lr, batch-size, epochs, dataset, model, and save-freq are consistent with what is used in train.py.

### Run Sample

After encountering challenges related to a significant drop in model accuracy post-watermark embedding, I have made a key modification to our training approach.
Previously, our approach to embedding watermarks involved training the model solely on watermark data in a separate phase from the main task training. This method, while effective in embedding the watermark significantly impacting its performance on the primary task. Instead of separately training the model on watermark data, I integrated the watermark directly into the training batches with the original dataset. This ensures the model learns both tasks simultaneously, maintaining high performance on its primary task while effectively embedding the watermark.

To show more information about the training and verification process, I would like to share log information with you below.

Firstly, I run the train code. That code successfully trains the model and increases the accuracy while adding model watermarking and generating Proof of Learning data.

```bash
(venv) PS C:\dev\PhD-Dissertation> python PoL/train.py --save-freq 100 --dataset CIFAR10 --model resnet20 --epochs 2
2024-03-31 22:51:11,377 - INFO - trying to allocate 0 gpus
2024-03-31 22:51:11,377 - INFO - Training started.
2024-03-31 22:51:11,377 - INFO - Using device: cpu
2024-03-31 22:51:11,951 - INFO - Loaded dataset 'CIFAR10' with 50000 samples.
2024-03-31 22:51:11,962 - INFO - Model architecture: resnet20
2024-03-31 22:51:11,962 - INFO - Optimizer: SGD
2024-03-31 22:57:56,140 - INFO - Verifying at step 3125
2024-03-31 22:58:09,044 - INFO - Accuracy: 45.79 %
2024-03-31 23:04:43,313 - INFO - Training completed.
2024-03-31 23:04:43,334 - INFO - Starting watermark validation.
2024-03-31 23:04:43,447 - INFO - Watermark validation accuracy: 100.00%
2024-03-31 23:04:58,705 - INFO - Accuracy: 58.77 %
2024-03-31 23:04:58,713 - INFO - Model with watermark saved at model_with_watermark.pth
2024-03-31 23:04:58,713 - INFO - Total time: 827.3360908031464
```
Secondly, I run the Proof of Learning Verification code. That code successfully verifies Proof of Learning and also verifies the watermark added at the training phase.

```bash
(venv) PS C:\dev\PhD-Dissertation> python PoL/verify.py --model-dir ./proof/CIFAR10_Batch100 --dist 1 2 inf cos --q 2
2024-03-31 23:32:11,715 - INFO - Starting the verification process...
2024-03-31 23:32:11,715 - INFO - Verifying model initialization...
2024-03-31 23:32:11,782 - INFO - The proof-of-learning passed the initialization verification.
2024-03-31 23:32:26,324 - INFO - Hash of the proof is valid.
2024-03-31 23:32:26,334 - INFO - Performing top-q verification...
2024-03-31 23:32:26,335 - INFO - Verifying epoch 1/2
2024-03-31 23:33:03,355 - INFO - Verifying epoch 2/2
2024-03-31 23:33:31,228 - INFO - Using device: cpu
2024-03-31 23:33:31,868 - INFO - Loaded dataset 'CIFAR10' with 50000 samples.
2024-03-31 23:33:31,883 - INFO - Model architecture: resnet20
2024-03-31 23:33:31,883 - INFO - Optimizer: SGD
2024-03-31 23:33:39,794 - INFO - Distance metric: 1 || threshold: 10000 || Q=2
2024-03-31 23:33:39,794 - INFO - Average top-q distance: 1501.8302001953125
2024-03-31 23:33:39,794 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-03-31 23:33:39,794 - INFO - Distance metric: 2 || threshold: 100 || Q=2
2024-03-31 23:33:39,796 - INFO - Average top-q distance: 5.058361053466797
2024-03-31 23:33:39,796 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-03-31 23:33:39,796 - INFO - Distance metric: inf || threshold: 1 || Q=2
2024-03-31 23:33:39,796 - INFO - Average top-q distance: 0.3334674760699272
2024-03-31 23:33:39,796 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-03-31 23:33:39,797 - INFO - Distance metric: cos || threshold: 0.1 || Q=2
2024-03-31 23:33:39,797 - INFO - Average top-q distance: 0.00472550094127655
2024-03-31 23:33:39,797 - INFO - None of the steps is above the threshold, the proof-of-learning is valid.
2024-03-31 23:33:39,799 - INFO - Verifying watermark presence in the model...
2024-03-31 23:33:39,921 - INFO - Watermark verification accuracy: 100.00%
2024-03-31 23:33:39,922 - INFO - Watermark verification successful: The watermark is present in the model.
```

### Spoof
To spoof a model on CIFAR-10 and CIFAR-100 with different attacks:
```
python spoof_cifar/attack.py --attack [1,2, or 3 for three attacks] --dataset ['CIFAR100' or 'CIFAR10'] --model [models defined in model.py] --t [spoof steps] --verify [1 or 0]
```
We use 'resnet20' for CIFAR-10 and 'resnet50' for CIFAR-100. t is the spoof steps, denoted by T in the paper, and here t =\frac{T}{100}.
We use '--cut' to fit different devices when 'cut' is set to 100, attack3 is same with attack2.

To spoof a model on the subset of ImageNet with different attacks:
```
python spoof_imagenet/spoof_imagenet.py --t [spoof steps] --verify [1 or 0]
python spoof_imagenet/spoof_attack3_imagenet.py --t [spoof steps] --verify [1 or 0]
```
'verify' is to control whether to verify the model.

### Model Generation
To train a model and create a proof-of-learning:
```
python PoL/train.py --save-freq [checkpointing interval] --dataset ['CIFAR100' or 'CIFAR10'] --model ['resnet50' or 'resnet20']
python spoof_imagenet/train.py --freq [checkpointing interval]
```
`save-freq` is a checkpointing interval, denoted by k in the paper [Proof-of-Learning: Definitions and Practice](https://arxiv.org/abs/2103.05633). 
Put the generated model in 'spoof_cifar/proof/[dataset]' to spoof the model. 
To generate CIFAR10 and CIFAR100 models with high accuracy:
```
python spoof_cifar/train.py --save-freq [checkpointing interval] --dataset ['CIFAR100' or 'CIFAR10'] --model ['resnet50' or 'resnet20']
```

To verify a given proof-of-learning or a given spoof:
```
python PoL/verify.py --model-dir [path/to/the/proof] --dist [distance metric] --q [query budget] --delta [slack parameter]
python spoof_imagenet/verify.py --k [checkpointing interval]
python spoof_cifar/verify.py --dataset ['CIFAR100' or 'CIFAR10'] --model [models defined in model.py] --iter [spoof steps * k] -- t [spoof steps] --k [checkpointing interval] 

```
Setting q to 0 or smaller will verify the whole proof, otherwise the top-q iterations for each epoch will be verified. More information about `q` and `delta` can be found in the paper. For `dist`, you could use one or more of `1`, `2`, `inf`, `cos` (if more than one, separate them by space). The first 3 are corresponding l_p norms, while `cos` is cosine distance. Note that if using more than one, the top-q iterations for all distance metrics will be verified.

Please ensure `lr`, `batch-sizr`, `epochs`, `dataset`, `model`, and `save-freq` are consistent with what is used in `train.py`.
