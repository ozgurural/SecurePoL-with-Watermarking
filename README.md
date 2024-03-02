## Enhancing Security of Proof-of-Learning against Spoofing Attacks Using Model Watermarking

**Abstract:** 
    The rapid advancement of machine learning (ML) technologies has underscored the imperative for robust security frameworks, especially in safeguarding the integrity and authenticity of ML model training processes. Proof-of-Learning (PoL), a mechanism designed to verify the computational labor invested in the training of ML models, stands at the forefront of addressing these security concerns. However, PoL systems face significant challenges, particularly from sophisticated spoofing attacks that undermine the foundational trust and reliability essential to ML applications. Concurrently, model watermarking emerges as a potent strategy for asserting model ownership and protecting intellectual property, offering a unique solution to enhance ML models' security against theft and unauthorized replication. This research delves into integrating PoL and model watermarking, proposing a synergistic approach to fortify ML models against various security threats. We establish a comprehensive, dual-layered verification architecture by embedding unique, discernible watermarks within models during training and meticulously documenting these alongside PoL proofs. This innovative methodology authenticates the computational effort through PoL and corroborates the model's authenticity and integrity via watermark detection, significantly amplifying defenses against potential spoofing. Such spoofing attempts often involve adversaries seeking to unduly replicate the computational trajectory and precisely mimic the watermark, posing a grave threat to model security. In exploring this integration, we tackle challenges, including maintaining watermark robustness, navigating the complexity of incorporating watermarking within PoL and balancing watermark security with model efficacy. Our systematic analysis of PoL vulnerabilities, juxtaposed with a tailored exploration of watermarking strategies for ML models, culminates in developing a provably secure PoL mechanism. Theoretical insights and empirical validations underscore the efficacy of merging model watermarking with PoL, markedly enhancing the framework's resilience to spoofing attacks. This significant stride towards the secure verification of ML models paves the way for further research to safeguard the integrity and reliability of model training across diverse ML applications, contributing to the overarching endeavor of securing ML models against an increasingly complex array of threats.

* **Project Advisor:** <br>
Dr. Kenji Yoshigoe `yoshigok@erau.edu` [email](mailto:yoshigok@erau.edu)<br>
https://scholar.google.com/citations?user=D6tC54MAAAAJ&hl=en

* **Phd. Student:** <br>
Ozgur Ural `uralo@my.erau.edu` [email](mailto:uralo@my.erau.edu)<br>
https://www.linkedin.com/in/uralozgur/


* **Embry-Riddle Aeronautical University, Daytona Beach**<br>
*Department of Electrical Engineering & Computer Science*<br>
Daytona Beach Campus<br>
1 Aerospace Boulevard<br>
Daytona Beach, FL 32114


# Proof-of-Learning

This repository is forked from the implementation of the paper used [Proof-of-Learning: Definitions and Practice](https://arxiv.org/abs/2103.05633), published in the 42nd IEEE Symposium on
Security and Privacy. In this paper, they introduced the concept of proof-of-learning in ML. Inspired by research on proof-of-work and verified computing, they observe how a seminal training algorithm, gradient descent, accumulates secret information due to its stochasticity. This produces a natural construction for a proof-of-learning, demonstrating that a party has expended the computation required to obtain a set of model parameters correctly. 

This approach has some problems regarding security against spoofing attacks, and our research is focusing how to make the PoL more secure against spoofing. 

The codebase is tested on two datasets: CIFAR-10 and CIFAR-100. 

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
Setting q to 0 or smaller will verify the whole proof, otherwise the top-q iterations for each epoch will be verified. More information about `q` and `delta` can be found in the paper. For `dist`, you could use one or more of `1`, `2`, `inf`, `cos` (if more than one, separate them by space). The first 3 are corresponding l_p norms, while `cos` is cosine distance. Note that if using more than one, the top-q iterations for all distance metrics will be verified.

Please ensure `lr`, `batch-sizr`, `epochs`, `dataset`, `model`, and `save-freq` are consistent with what is used in `train.py`.


### Run Sample 

```shell
PS C:\dev\phd-2024\Proof-of-Learning> python -m venv venv
PS C:\dev\phd-2024\Proof-of-Learning> .\venv\Scripts\activate
(venv) PS C:\dev\phd-2024\Proof-of-Learning> pip install torch==1.8.0 torchvision==0.9.0 numpy scipy 
(venv) PS C:\dev\phd-2024\Proof-of-Learning> python ./train.py --save-freq 100 --dataset CIFAR10 --model resnet20
trying to allocate 0 gpus
Accuracy: 45.37 %
(venv) PS C:\dev\phd-2024\Proof-of-Learning>  python ./verify.py --model-dir ./proof/CIFAR10_test --dist 1 2 inf cos --q 2
The proof-of-learning passed the initialization verification.
Hash of the proof is valid.
Verifying epoch 1/2
Distance metric: 1 || threshold: 1000 || Q=2
Average top-q distance: 541.6904602050781
None of the steps is above the threshold, the proof-of-learning is valid.
Distance metric: 2 || threshold: 10 || Q=2
Average top-q distance: 1.6582163572311401
None of the steps is above the threshold, the proof-of-learning is valid.
Distance metric: inf || threshold: 0.1 || Q=2
Average top-q distance: 0.08480853959918022
1 / 2 (50.0%) of the steps are above the threshold, the proof-of-learning is invalid.
Distance metric: cos || threshold: 0.01 || Q=2
Average top-q distance: 0.0006642043590545654
None of the steps is above the threshold, the proof-of-learning is valid.
Verifying epoch 2/2
Distance metric: 1 || threshold: 1000 || Q=2
Average top-q distance: 410.57948303222656
None of the steps is above the threshold, the proof-of-learning is valid.
Distance metric: 2 || threshold: 10 || Q=2
Average top-q distance: 1.234419047832489
None of the steps is above the threshold, the proof-of-learning is valid.
Distance metric: inf || threshold: 0.1 || Q=2
Average top-q distance: 0.04239170253276825
None of the steps is above the threshold, the proof-of-learning is valid.
Distance metric: cos || threshold: 0.01 || Q=2
Average top-q distance: 0.00030875205993652344
None of the steps is above the threshold, the proof-of-learning is valid.
```
