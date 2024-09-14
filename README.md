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
(venv) PS C:\dev\SecurePoL-Watermarking> python PoL/train.py --save-freq 100 --dataset CIFAR10 --model resnet20 --epochs 5
2024-09-14 12:31:38,834 - INFO - Trying to allocate 0 GPUs
2024-09-14 12:31:38,834 - INFO - Using device: cpu
2024-09-14 12:31:39,284 - INFO - Loaded dataset 'CIFAR10' with 50000 samples.
2024-09-14 12:31:39,284 - INFO - Generated training sequence with length 250000
2024-09-14 12:31:53,160 - INFO - Model architecture: resnet20
2024-09-14 12:31:53,160 - INFO - Optimizer: SGD
2024-09-14 12:31:53,168 - INFO - Saved initial model checkpoint at step 0
2024-09-14 12:31:53,168 - INFO - Starting epoch 1/5
2024-09-14 12:32:10,883 - INFO - Feature-based watermark embedded at step 0
2024-09-14 12:32:56,733 - INFO - Saved checkpoint at step 100
2024-09-14 12:32:56,953 - INFO - Feature-based watermark embedded at step 100
2024-09-14 12:33:47,422 - INFO - Saved checkpoint at step 200
2024-09-14 12:33:47,727 - INFO - Feature-based watermark embedded at step 200
2024-09-14 12:34:40,980 - INFO - Saved checkpoint at step 300
2024-09-14 12:34:41,290 - INFO - Feature-based watermark embedded at step 300
2024-09-14 12:35:28,134 - INFO - Saved checkpoint at step 400
2024-09-14 12:35:28,363 - INFO - Feature-based watermark embedded at step 400
2024-09-14 12:36:12,558 - INFO - Saved checkpoint at step 500
2024-09-14 12:36:12,804 - INFO - Feature-based watermark embedded at step 500
2024-09-14 12:36:58,993 - INFO - Saved checkpoint at step 600
2024-09-14 12:36:59,228 - INFO - Feature-based watermark embedded at step 600
2024-09-14 12:37:44,592 - INFO - Saved checkpoint at step 700
2024-09-14 12:37:44,844 - INFO - Feature-based watermark embedded at step 700
2024-09-14 12:38:29,762 - INFO - Saved checkpoint at step 800
2024-09-14 12:38:29,997 - INFO - Feature-based watermark embedded at step 800
2024-09-14 12:39:33,225 - INFO - Saved checkpoint at step 900
2024-09-14 12:39:33,533 - INFO - Feature-based watermark embedded at step 900
2024-09-14 12:40:34,921 - INFO - Saved checkpoint at step 1000
2024-09-14 12:40:35,234 - INFO - Feature-based watermark embedded at step 1000
2024-09-14 12:41:35,811 - INFO - Saved checkpoint at step 1100
2024-09-14 12:41:36,120 - INFO - Feature-based watermark embedded at step 1100
2024-09-14 12:42:33,958 - INFO - Saved checkpoint at step 1200
2024-09-14 12:42:34,258 - INFO - Feature-based watermark embedded at step 1200
2024-09-14 12:43:32,630 - INFO - Saved checkpoint at step 1300
2024-09-14 12:43:32,930 - INFO - Feature-based watermark embedded at step 1300
2024-09-14 12:44:30,826 - INFO - Saved checkpoint at step 1400
2024-09-14 12:44:31,148 - INFO - Feature-based watermark embedded at step 1400
2024-09-14 12:45:28,630 - INFO - Saved checkpoint at step 1500
2024-09-14 12:45:29,019 - INFO - Feature-based watermark embedded at step 1500
2024-09-14 12:46:26,584 - INFO - Saved checkpoint at step 1600
2024-09-14 12:46:26,881 - INFO - Feature-based watermark embedded at step 1600
2024-09-14 12:47:25,259 - INFO - Saved checkpoint at step 1700
2024-09-14 12:47:25,566 - INFO - Feature-based watermark embedded at step 1700
2024-09-14 12:48:22,599 - INFO - Saved checkpoint at step 1800
2024-09-14 12:48:22,900 - INFO - Feature-based watermark embedded at step 1800
2024-09-14 12:49:20,254 - INFO - Saved checkpoint at step 1900
2024-09-14 12:49:20,540 - INFO - Feature-based watermark embedded at step 1900
2024-09-14 12:50:19,234 - INFO - Validation Accuracy: 52.06%
2024-09-14 12:50:19,236 - INFO - Starting epoch 2/5
2024-09-14 12:51:11,659 - INFO - Saved checkpoint at step 2000
2024-09-14 12:51:11,953 - INFO - Feature-based watermark embedded at step 2000
2024-09-14 12:52:09,288 - INFO - Saved checkpoint at step 2100
2024-09-14 12:52:09,590 - INFO - Feature-based watermark embedded at step 2100
2024-09-14 12:53:06,994 - INFO - Saved checkpoint at step 2200
2024-09-14 12:53:07,286 - INFO - Feature-based watermark embedded at step 2200
2024-09-14 12:54:04,648 - INFO - Saved checkpoint at step 2300
2024-09-14 12:54:04,956 - INFO - Feature-based watermark embedded at step 2300
2024-09-14 12:55:02,256 - INFO - Saved checkpoint at step 2400
2024-09-14 12:55:02,567 - INFO - Feature-based watermark embedded at step 2400
2024-09-14 12:56:01,349 - INFO - Saved checkpoint at step 2500
2024-09-14 12:56:01,640 - INFO - Feature-based watermark embedded at step 2500
2024-09-14 12:57:01,263 - INFO - Saved checkpoint at step 2600
2024-09-14 12:57:01,572 - INFO - Feature-based watermark embedded at step 2600
2024-09-14 12:57:59,816 - INFO - Saved checkpoint at step 2700
2024-09-14 12:58:00,116 - INFO - Feature-based watermark embedded at step 2700
2024-09-14 12:58:55,288 - INFO - Saved checkpoint at step 2800
2024-09-14 12:58:55,619 - INFO - Feature-based watermark embedded at step 2800
2024-09-14 12:59:51,625 - INFO - Saved checkpoint at step 2900
2024-09-14 12:59:51,921 - INFO - Feature-based watermark embedded at step 2900
2024-09-14 13:00:47,993 - INFO - Saved checkpoint at step 3000
2024-09-14 13:00:48,291 - INFO - Feature-based watermark embedded at step 3000
2024-09-14 13:01:44,702 - INFO - Saved checkpoint at step 3100
2024-09-14 13:01:44,948 - INFO - Feature-based watermark embedded at step 3100
2024-09-14 13:02:39,836 - INFO - Saved checkpoint at step 3200
2024-09-14 13:02:40,072 - INFO - Feature-based watermark embedded at step 3200
2024-09-14 13:03:32,057 - INFO - Saved checkpoint at step 3300
2024-09-14 13:03:32,312 - INFO - Feature-based watermark embedded at step 3300
2024-09-14 13:04:23,148 - INFO - Saved checkpoint at step 3400
2024-09-14 13:04:23,417 - INFO - Feature-based watermark embedded at step 3400
2024-09-14 13:05:14,814 - INFO - Saved checkpoint at step 3500
2024-09-14 13:05:15,097 - INFO - Feature-based watermark embedded at step 3500
2024-09-14 13:06:03,893 - INFO - Saved checkpoint at step 3600
2024-09-14 13:06:04,125 - INFO - Feature-based watermark embedded at step 3600
2024-09-14 13:06:51,616 - INFO - Saved checkpoint at step 3700
2024-09-14 13:06:51,876 - INFO - Feature-based watermark embedded at step 3700
2024-09-14 13:07:47,456 - INFO - Saved checkpoint at step 3800
2024-09-14 13:07:47,668 - INFO - Feature-based watermark embedded at step 3800
2024-09-14 13:08:34,516 - INFO - Saved checkpoint at step 3900
2024-09-14 13:08:34,736 - INFO - Feature-based watermark embedded at step 3900
2024-09-14 13:09:00,046 - INFO - Validation Accuracy: 57.15%
2024-09-14 13:09:00,056 - INFO - Starting epoch 3/5
2024-09-14 13:10:08,896 - INFO - Saved checkpoint at step 4000
2024-09-14 13:10:09,108 - INFO - Feature-based watermark embedded at step 4000
2024-09-14 13:10:56,702 - INFO - Saved checkpoint at step 4100
2024-09-14 13:10:56,929 - INFO - Feature-based watermark embedded at step 4100
2024-09-14 13:11:43,955 - INFO - Saved checkpoint at step 4200
2024-09-14 13:11:44,177 - INFO - Feature-based watermark embedded at step 4200
2024-09-14 13:12:32,406 - INFO - Saved checkpoint at step 4300
2024-09-14 13:12:32,807 - INFO - Feature-based watermark embedded at step 4300
2024-09-14 13:13:25,049 - INFO - Saved checkpoint at step 4400
2024-09-14 13:13:25,344 - INFO - Feature-based watermark embedded at step 4400
2024-09-14 13:14:11,979 - INFO - Saved checkpoint at step 4500
2024-09-14 13:14:12,214 - INFO - Feature-based watermark embedded at step 4500
2024-09-14 13:14:57,420 - INFO - Saved checkpoint at step 4600
2024-09-14 13:14:57,638 - INFO - Feature-based watermark embedded at step 4600
2024-09-14 13:15:53,555 - INFO - Saved checkpoint at step 4700
2024-09-14 13:15:53,810 - INFO - Feature-based watermark embedded at step 4700
2024-09-14 13:16:43,064 - INFO - Saved checkpoint at step 4800
2024-09-14 13:16:43,284 - INFO - Feature-based watermark embedded at step 4800
2024-09-14 13:17:31,579 - INFO - Saved checkpoint at step 4900
2024-09-14 13:17:31,804 - INFO - Feature-based watermark embedded at step 4900
2024-09-14 13:18:16,219 - INFO - Saved checkpoint at step 5000
2024-09-14 13:18:16,446 - INFO - Feature-based watermark embedded at step 5000
2024-09-14 13:19:08,236 - INFO - Saved checkpoint at step 5100
2024-09-14 13:19:08,486 - INFO - Feature-based watermark embedded at step 5100
2024-09-14 13:19:57,684 - INFO - Saved checkpoint at step 5200
2024-09-14 13:19:57,934 - INFO - Feature-based watermark embedded at step 5200
2024-09-14 13:20:43,482 - INFO - Saved checkpoint at step 5300
2024-09-14 13:20:43,729 - INFO - Feature-based watermark embedded at step 5300
2024-09-14 13:21:28,684 - INFO - Saved checkpoint at step 5400
2024-09-14 13:21:28,901 - INFO - Feature-based watermark embedded at step 5400
2024-09-14 13:22:14,723 - INFO - Saved checkpoint at step 5500
2024-09-14 13:22:14,951 - INFO - Feature-based watermark embedded at step 5500
2024-09-14 13:23:06,271 - INFO - Saved checkpoint at step 5600
2024-09-14 13:23:06,507 - INFO - Feature-based watermark embedded at step 5600
2024-09-14 13:23:54,439 - INFO - Saved checkpoint at step 5700
2024-09-14 13:23:54,756 - INFO - Feature-based watermark embedded at step 5700
2024-09-14 13:24:44,668 - INFO - Saved checkpoint at step 5800
2024-09-14 13:24:44,913 - INFO - Feature-based watermark embedded at step 5800
2024-09-14 13:25:37,359 - INFO - Validation Accuracy: 60.86%
2024-09-14 13:25:37,363 - INFO - Starting epoch 4/5
2024-09-14 13:26:12,152 - INFO - Saved checkpoint at step 5900
2024-09-14 13:26:12,380 - INFO - Feature-based watermark embedded at step 5900
2024-09-14 13:26:58,113 - INFO - Saved checkpoint at step 6000
2024-09-14 13:26:58,348 - INFO - Feature-based watermark embedded at step 6000
2024-09-14 13:27:44,282 - INFO - Saved checkpoint at step 6100
2024-09-14 13:27:44,512 - INFO - Feature-based watermark embedded at step 6100
2024-09-14 13:28:31,757 - INFO - Saved checkpoint at step 6200
2024-09-14 13:28:31,992 - INFO - Feature-based watermark embedded at step 6200
2024-09-14 13:29:18,272 - INFO - Saved checkpoint at step 6300
2024-09-14 13:29:18,575 - INFO - Feature-based watermark embedded at step 6300
2024-09-14 13:30:07,750 - INFO - Saved checkpoint at step 6400
2024-09-14 13:30:07,992 - INFO - Feature-based watermark embedded at step 6400
2024-09-14 13:31:15,103 - INFO - Saved checkpoint at step 6500
2024-09-14 13:31:15,629 - INFO - Feature-based watermark embedded at step 6500
2024-09-14 13:32:11,818 - INFO - Saved checkpoint at step 6600
2024-09-14 13:32:12,088 - INFO - Feature-based watermark embedded at step 6600
2024-09-14 13:33:00,543 - INFO - Saved checkpoint at step 6700
2024-09-14 13:33:00,801 - INFO - Feature-based watermark embedded at step 6700
2024-09-14 13:33:52,344 - INFO - Saved checkpoint at step 6800
2024-09-14 13:33:52,596 - INFO - Feature-based watermark embedded at step 6800
2024-09-14 13:34:44,419 - INFO - Saved checkpoint at step 6900
2024-09-14 13:34:44,676 - INFO - Feature-based watermark embedded at step 6900
2024-09-14 13:35:37,156 - INFO - Saved checkpoint at step 7000
2024-09-14 13:35:37,413 - INFO - Feature-based watermark embedded at step 7000
2024-09-14 13:36:38,067 - INFO - Saved checkpoint at step 7100
2024-09-14 13:36:38,332 - INFO - Feature-based watermark embedded at step 7100
2024-09-14 13:37:30,063 - INFO - Saved checkpoint at step 7200
2024-09-14 13:37:30,325 - INFO - Feature-based watermark embedded at step 7200
2024-09-14 13:38:22,089 - INFO - Saved checkpoint at step 7300
2024-09-14 13:38:22,337 - INFO - Feature-based watermark embedded at step 7300
2024-09-14 13:39:17,032 - INFO - Saved checkpoint at step 7400
2024-09-14 13:39:17,278 - INFO - Feature-based watermark embedded at step 7400
2024-09-14 13:40:15,353 - INFO - Saved checkpoint at step 7500
2024-09-14 13:40:15,636 - INFO - Feature-based watermark embedded at step 7500
2024-09-14 13:41:35,921 - INFO - Saved checkpoint at step 7600
2024-09-14 13:41:36,155 - INFO - Feature-based watermark embedded at step 7600
2024-09-14 13:42:26,265 - INFO - Saved checkpoint at step 7700
2024-09-14 13:42:26,516 - INFO - Feature-based watermark embedded at step 7700
2024-09-14 13:43:18,815 - INFO - Saved checkpoint at step 7800
2024-09-14 13:43:19,066 - INFO - Feature-based watermark embedded at step 7800
2024-09-14 13:43:48,171 - INFO - Validation Accuracy: 63.31%
2024-09-14 13:43:48,174 - INFO - Starting epoch 5/5
2024-09-14 13:44:46,190 - INFO - Saved checkpoint at step 7900
2024-09-14 13:44:46,435 - INFO - Feature-based watermark embedded at step 7900
2024-09-14 13:45:40,053 - INFO - Saved checkpoint at step 8000
2024-09-14 13:45:40,571 - INFO - Feature-based watermark embedded at step 8000
2024-09-14 13:46:34,226 - INFO - Saved checkpoint at step 8100
2024-09-14 13:46:34,521 - INFO - Feature-based watermark embedded at step 8100
2024-09-14 13:47:33,981 - INFO - Saved checkpoint at step 8200
2024-09-14 13:47:34,250 - INFO - Feature-based watermark embedded at step 8200
2024-09-14 13:48:37,720 - INFO - Saved checkpoint at step 8300
2024-09-14 13:48:38,116 - INFO - Feature-based watermark embedded at step 8300
2024-09-14 13:49:53,794 - INFO - Saved checkpoint at step 8400
2024-09-14 13:49:54,136 - INFO - Feature-based watermark embedded at step 8400
2024-09-14 13:51:03,513 - INFO - Saved checkpoint at step 8500
2024-09-14 13:51:03,885 - INFO - Feature-based watermark embedded at step 8500
2024-09-14 13:52:15,231 - INFO - Saved checkpoint at step 8600
2024-09-14 13:52:15,713 - INFO - Feature-based watermark embedded at step 8600
2024-09-14 13:53:44,807 - INFO - Saved checkpoint at step 8700
2024-09-14 13:53:45,202 - INFO - Feature-based watermark embedded at step 8700
2024-09-14 13:55:06,887 - INFO - Saved checkpoint at step 8800
2024-09-14 13:55:07,212 - INFO - Feature-based watermark embedded at step 8800
2024-09-14 13:56:06,366 - INFO - Saved checkpoint at step 8900
2024-09-14 13:56:06,694 - INFO - Feature-based watermark embedded at step 8900
2024-09-14 13:57:06,522 - INFO - Saved checkpoint at step 9000
2024-09-14 13:57:06,831 - INFO - Feature-based watermark embedded at step 9000
2024-09-14 13:58:06,942 - INFO - Saved checkpoint at step 9100
2024-09-14 13:58:07,259 - INFO - Feature-based watermark embedded at step 9100
2024-09-14 13:59:12,553 - INFO - Saved checkpoint at step 9200
2024-09-14 13:59:12,905 - INFO - Feature-based watermark embedded at step 9200
2024-09-14 14:00:20,727 - INFO - Saved checkpoint at step 9300
2024-09-14 14:00:21,074 - INFO - Feature-based watermark embedded at step 9300
2024-09-14 14:01:27,413 - INFO - Saved checkpoint at step 9400
2024-09-14 14:01:27,729 - INFO - Feature-based watermark embedded at step 9400
2024-09-14 14:02:26,639 - INFO - Saved checkpoint at step 9500
2024-09-14 14:02:26,957 - INFO - Feature-based watermark embedded at step 9500
2024-09-14 14:03:26,957 - INFO - Saved checkpoint at step 9600
2024-09-14 14:03:27,303 - INFO - Feature-based watermark embedded at step 9600
2024-09-14 14:04:25,881 - INFO - Saved checkpoint at step 9700
2024-09-14 14:04:26,203 - INFO - Feature-based watermark embedded at step 9700
2024-09-14 14:05:36,847 - INFO - Validation Accuracy: 65.42%
2024-09-14 14:05:36,874 - INFO - Saved final model checkpoint at step 9770
2024-09-14 14:05:36,889 - INFO - Starting feature-based watermark validation.
2024-09-14 14:05:37,030 - INFO - Feature-based watermark validation successful: Watermark detected.
2024-09-14 14:05:37,030 - INFO - Watermark Detection Accuracy: 100.00%
2024-09-14 14:06:05,886 - INFO - Validation Accuracy: 65.42%
2024-09-14 14:06:05,903 - INFO - Model with watermark saved at model_with_watermark.pth
2024-09-14 14:06:05,903 - INFO - Total time: 5667.07 seconds
```
Secondly, I run the Proof of Learning Verification code. That code verifies proof of learning and the watermark added during the training phase.

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
