
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
