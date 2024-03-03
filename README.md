# Enhancing Security of Proof-of-Learning against Spoofing Attacks Using Model Watermarking

**Abstract:**  
The rapid advancement of machine learning (ML) technologies has underscored the imperative for robust security frameworks, especially in safeguarding the integrity and authenticity of ML model training processes. Proof-of-Learning (PoL), a mechanism designed to verify the computational labor invested in the training of ML models, stands at the forefront of addressing these security concerns. However, PoL systems face significant challenges, particularly from sophisticated spoofing attacks that undermine the foundational trust and reliability essential to ML applications. Concurrently, model watermarking emerges as a potent strategy for asserting model ownership and protecting intellectual property, offering a unique solution to enhance ML models' security against theft and unauthorized replication. This research delves into integrating PoL and model watermarking, proposing a synergistic approach to fortify ML models against various security threats. We establish a comprehensive, dual-layered verification architecture by embedding unique, discernible watermarks within models during training and meticulously documenting these alongside PoL proofs. This innovative methodology authenticates the computational effort through PoL and corroborates the model's authenticity and integrity via watermark detection, significantly amplifying defenses against potential spoofing. Such spoofing attempts often involve adversaries seeking to unduly replicate the computational trajectory and precisely mimic the watermark, posing a grave threat to model security. In exploring this integration, we tackle challenges, including maintaining watermark robustness, navigating the complexity of incorporating watermarking within PoL and balancing watermark security with model efficacy. Our systematic analysis of PoL vulnerabilities, juxtaposed with a tailored exploration of watermarking strategies for ML models, culminates in developing a provably secure PoL mechanism. Theoretical insights and empirical validations underscore the efficacy of merging model watermarking with PoL, markedly enhancing the framework's resilience to spoofing attacks. This significant stride towards the secure verification of ML models paves the way for further research to safeguard the integrity and reliability of model training across diverse ML applications, contributing to the overarching endeavor of securing ML models against an increasingly complex array of threats.

**Project Advisor:**  
Dr. Kenji Yoshigoe  
Email: [yoshigok@erau.edu](mailto:yoshigok@erau.edu)  
[Google Scholar Profile](https://scholar.google.com/citations?user=D6tC54MAAAAJ&hl=en)

**PhD. Student:**  
Ozgur Ural  
Email: [uralo@my.erau.edu](mailto:uralo@my.erau.edu)  
[LinkedIn Profile](https://www.linkedin.com/in/uralozgur/)

**Embry-Riddle Aeronautical University, Daytona Beach**  
*Department of Electrical Engineering & Computer Science*  
Daytona Beach Campus  
1 Aerospace Boulevard  
Daytona Beach, FL 32114

## Enhancements in Proof-of-Learning with Adversarial Examples

This repository stems from the work presented in ["Proof-of-Learning: Definitions and Practice"](https://arxiv.org/abs/2103.05633), which was featured in the 42nd IEEE Symposium on Security and Privacy. The paper pioneers the Proof-of-Learning (PoL) concept within machine learning (ML), drawing inspiration from the proof-of-work systems and verified computing. It highlights how gradient descent, a fundamental algorithm in training ML models, inherently captures stochastic information. This characteristic facilitates the generation of PoL, substantiating that the computational efforts to derive a model's parameters were legitimately executed.

However, this system could be more impervious to security challenges, notably spoofing attacks, which pose a significant risk to its integrity. Our research is dedicated to advancing the robustness of PoL, specifically by developing strategies to bolster its resilience against such attacks.

In addition to the original PoL framework, this repository incorporates insights from the study on ["Adversarial Examples for Proof-of-Learning"](https://arxiv.org/abs/2108.09454). This paper elucidates a methodology that exposes vulnerabilities within the PoL by deploying adversarial examples, hence challenging the security assumptions of the initial PoL proposition. Our work here is to address these security concerns and fortify the PoL against these identified threats.

We have employed datasets such as CIFAR-10, CIFAR-100, and a subset of ImageNet to test and validate our findings. For an in-depth understanding of our contributions to the PoL framework and the adversarial techniques, we would like you to take a look at the referenced papers.

### Dependency
Our code is implemented and tested on PyTorch. The following packages are used:
\subsection*{Dependency}
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

```bash
PS C:\dev\PhD-Dissertation> python -m venv venv
PS C:\dev\PhD-Dissertation> .\venv\Scripts\activate
(venv) PS C:\dev\PhD-Dissertation> pip install torch==1.8.0 torchvision==0.9.0 numpy scipy requests
(venv) PS C:\dev\PhD-Dissertation> python PoL/train.py --save-freq 100 --dataset CIFAR10 --model resnet20 --epochs 5
trying to allocate 1 gpus
Epoch 1
Accuracy: 47.83 %
Epoch 2
Accuracy: 61.65 %
Epoch 3
Accuracy: 70.57 %
Epoch 4
Accuracy: 69.78 %
Total time:  1822.553815126419
Accuracy: 68.68 %
(venv) PS C:\dev\phd-2024\Proof-of-Learning>  python PoL/verify.py --model-dir ./proof/CIFAR10_Batch100 --dist 1 2 inf cos --q 0
Distance metric: 1 || threshold: 1000
Average distance: 1312.5277099609375, Max distance: 1312.5277099609375, Min distance: 1312.5277099609375
1 / 1 (100.0%) of the steps are above the threshold, the proof-of-learning is invalid.
Distance metric: 2 || threshold: 10
Average distance: 3.9775619506835938, Max distance: 3.9775619506835938, Min distance: 3.9775619506835938
None of the steps is above the threshold, the proof-of-learning is valid.
Distance metric: inf || threshold: 0.1
Average distance: 0.13681060075759888, Max distance: 0.13681060075759888, Min distance: 0.13681060075759888
1 / 1 (100.0%) of the steps are above the threshold, the proof-of-learning is invalid.
Distance metric: cos || threshold: 0.01
Average distance: 0.0038004517555236816, Max distance: 0.0038004517555236816, Min distance: 0.0038004517555236816
None of the steps is above the threshold, the proof-of-learning is valid.
(myenv) PS C:\dev\Adversarial-examples-for-Proof-of-Learning>
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
`save-freq` is a checkpointing interval, denoted by k in the paper[Proof-of-Learning: Definitions and Practice](https://arxiv.org/abs/2103.05633). 
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
