# Proof-of-Learning

This repository is forked from the implementation of the paper used [Proof-of-Learning: Definitions and Practice](https://arxiv.org/abs/2103.05633), published in the 42nd IEEE Symposium on
Security and Privacy. In this paper, they introduced the concept of proof-of-learning in ML. Inspired by research on proof-of-work and verified computing, they observe how a seminal training algorithm, gradient descent, accumulates secret information due to its stochasticity. This produces a natural construction for a proof-of-learning, which demonstrates that a party has expended the computation required to obtain a set of model parameters correctly. 

The code is tested on two datasets: CIFAR-10, and CIFAR-100. 

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
`save-freq` is checkpointing interval, denoted by k in the paper. There are a few other arguments that you could find at the end of the script. 

Note that the proposed algorithm does not interact with the training process, so it could be applied to any kinds of gradient-descent based models.


### Verify
To verify a given proof-of-learning:
```
python verify.py --model-dir [path/to/the/proof] --dist [distance metric] --q [query budget] --delta [slack parameter]
```
Setting q to 0 or smaller will verify the whole proof, otherwise the top-q iterations for each epoch will be verified. More information about `q` and `delta` can be found in the paper. For `dist`, you could use one or more of `1`, `2`, `inf`, `cos` (if more than one, separate them by space). The first 3 are corresponding l_p norms, while `cos` is cosine distance. Note that if using more than one, the top-q iterations in terms of all distance metrics will be verified.

Please make sure `lr`, `batch-sizr`, `epochs`, `dataset`, `model`, and `save-freq` are consistent with what used in `train.py`.


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
