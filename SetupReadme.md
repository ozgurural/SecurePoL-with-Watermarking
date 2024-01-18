
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
