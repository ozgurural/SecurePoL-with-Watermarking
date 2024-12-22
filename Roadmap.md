

**To-Do List for Experiments: Enhancing Security of Proof-of-Learning with Feature-Based Model Watermarking**

**Objective:** Conduct a series of experiments to evaluate the impact of feature-based model watermarking on model accuracy, watermark robustness, and security against spoofing attacks. The experiments will compare models trained without watermarking (baseline) and with watermarking using different values of `k` (watermark embedding frequency).

---

### **1. Prepare the Experimental Environment**

- **1.1** **Set Up the Environment**

  - **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    ```
  - **Activate the virtual environment**:
    - On Linux/macOS:
      ```bash
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
  - **Install required packages**:
    ```bash
    pip install torch==1.8.0 torchvision==0.9.0 numpy scipy
    ```

- **1.2** **Clone the Updated Repository**

  ```bash
  git clone https://github.com/ozgurural/SecurePoL-Watermarking.git
  ```
  - Navigate to the project directory:
    ```bash
    cd SecurePoL-Watermarking
    ```

- **1.3** **Verify Codebase**

  - Ensure you have the latest updates:
    ```bash
    git pull origin main
    ```

---

### **2. Conduct Baseline Experiment (Without Watermarking)**

**Objective:** Establish the baseline model performance without any watermarking.

- **2.1** **Train the Baseline Model**

  - **Command:**
    ```bash
    python PoL/train.py --save-freq 100 --dataset CIFAR10 --model resnet20 --epochs 5
    ```
    - This trains a ResNet-20 model on CIFAR-10 for 5 epochs, saving checkpoints every 100 steps.

- **2.2** **Evaluate Baseline Model Performance**

  - **Metrics to Record:**
    - Validation accuracy
    - Precision
    - Recall
    - F1-score
  - **Note:** The training script outputs the validation accuracy at the end.

- **2.3** **Verify Proof-of-Learning for Baseline Model**

  - **Command:**
    ```bash
    python PoL/verify.py --model-dir proof/CIFAR10_Batch100 --dataset CIFAR10 --model resnet20 --epochs 5 --save-freq 100 --batch-size 128 --lr 0.1 --dist 1 2 inf cos --delta 10000 100 1 0.1
    ```
    - Replace `proof/CIFAR10_Batch100` with the actual path to your proof directory if different.
    - Ensure parameters match those used during training.

---

### **3. Experiment with Watermarking (Varying `k`)**

**Objective:** Evaluate the impact of watermarking on model performance and watermark effectiveness for different values of `k`.

- **3.1** **Define Values of `k` to Test**

  - Suggested values:
    ```python
    k_values = [500, 1000, 2000, 5000]
    ```
    - These represent the watermark embedding frequency.

- **3.2** **For Each Value of `k`, Perform the Following Steps:**

**Loop Over `k_values`:**

#### **3.2.1** **Train the Model with Watermarking**

- **Command:**
  ```bash
  python PoL/train_with_watermark.py --save-freq 100 --dataset CIFAR10 --model resnet20 --epochs 5 --lambda-wm 0.01 --k [K_VALUE] --watermark-key 'secret_key'
  ```
  - Replace `[K_VALUE]` with the current value from `k_values`.
  - This trains the model with feature-based watermarking.

#### **3.2.2** **Evaluate Model Performance**

- **Metrics to Record:**
  - Validation accuracy
  - Precision
  - Recall
  - F1-score

#### **3.2.3** **Verify Proof-of-Learning and Watermark Presence**

- **Command:**
  ```bash
  python PoL/verify.py --model-dir proof/CIFAR10_Batch100_k[K_VALUE] --dataset CIFAR10 --model resnet20 --epochs 5 --save-freq 100 --batch-size 128 --lr 0.1 --lambda-wm 0.01 --k [K_VALUE] --watermark-key 'secret_key' --dist 1 2 inf cos --delta 10000 100 1 0.1 --watermark-path model_with_watermark_k[K_VALUE].pth
  ```
  - Replace `[K_VALUE]` accordingly.
  - Ensure that `--model-dir` and `--watermark-path` point to the correct directories and files for each `k`.

#### **3.2.4** **Record the Results**

- **Document:**
  - Model performance metrics.
  - Whether the watermark was successfully detected.
  - Any observations about training time or convergence.

#### **3.2.5** **Analyze Impact of `k` on Performance and Watermark Robustness**

- **Tasks:**
  - Plot accuracy vs. `k`.
  - Analyze how increasing `k` affects model performance.
  - Discuss trade-offs observed.

---

### **4. Evaluate Watermark Effectiveness Under Attacks**

**Objective:** Assess the robustness of the watermark against model fine-tuning and pruning.

#### **4.1** **Fine-Tuning the Watermarked Model**

- **4.1.1** **Fine-Tune the Model**

  - **Command:**
    ```bash
    python PoL/fine_tune.py --model-path model_with_watermark_k[K_VALUE].pth --dataset CIFAR10 --model resnet20 --epochs 2
    ```
    - This fine-tunes the watermarked model for 2 additional epochs without the watermark loss.

- **4.1.2** **Verify Watermark Post Fine-Tuning**

  - **Command:**
    ```bash
    python PoL/verify.py --model-dir fine_tuned_model_k[K_VALUE] --dataset CIFAR10 --model resnet20 --epochs 5 --save-freq 100 --batch-size 128 --lr 0.1 --lambda-wm 0 --k [K_VALUE] --watermark-key 'secret_key' --dist 1 2 inf cos --delta 10000 100 1 0.1 --watermark-path fine_tuned_model_k[K_VALUE].pth
    ```
    - Check if the watermark is still detectable.

- **4.1.3** **Document Findings**

  - Note if the watermark survived fine-tuning.
  - Record any changes in model performance.

#### **4.2** **Pruning the Watermarked Model**

- **4.2.1** **Apply Model Pruning**

  - Implement a pruning script or use an existing one to prune weights.
  - **Example Command:**
    ```bash
    python PoL/prune_model.py --model-path model_with_watermark_k[K_VALUE].pth --prune-percentage [PERCENTAGE]
    ```
    - Replace `[PERCENTAGE]` with the desired pruning level (e.g., 10%).

- **4.2.2** **Verify Watermark Post Pruning**

  - **Command:**
    ```bash
    python PoL/verify.py --model-dir pruned_model_k[K_VALUE] --dataset CIFAR10 --model resnet20 --epochs 5 --save-freq 100 --batch-size 128 --lr 0.1 --lambda-wm 0 --k [K_VALUE] --watermark-key 'secret_key' --dist 1 2 inf cos --delta 10000 100 1 0.1 --watermark-path pruned_model_k[K_VALUE].pth
    ```
    - Assess if the watermark remains detectable after pruning.

- **4.2.3** **Document Findings**

  - Record watermark detection results post pruning.
  - Note any performance changes.

---

### **5. Compare with Other Watermarking Techniques**

**Objective:** Compare feature-based watermarking with parameter perturbation-based and non-intrusive watermarking methods.

#### **5.1** **Implement and Test Parameter Perturbation-Based Watermarking**

- **If Not Implemented:** Develop a training script `train_with_param_watermark.py` that modifies model parameters directly for watermarking.

- **5.1.1** **Train Models with Parameter Perturbation**

  - **Command:**
    ```bash
    python PoL/train_with_param_watermark.py --save-freq 100 --dataset CIFAR10 --model resnet20 --epochs 5 --lambda-wm 0.01 --k [K_VALUE] --watermark-key 'secret_key'
    ```
    - Use the same `k_values` as before.

- **5.1.2** **Evaluate and Verify**

  - Follow similar evaluation and verification steps as in **Section 3**.

#### **5.2** **Implement and Test Non-Intrusive Watermarking**

- **If Not Implemented:** Develop a script `train_with_nonintrusive_watermark.py` for non-intrusive watermarking (e.g., using special trigger inputs).

- **5.2.1** **Train Models with Non-Intrusive Watermarking**

  - **Command:**
    ```bash
    python PoL/train_with_nonintrusive_watermark.py --save-freq 100 --dataset CIFAR10 --model resnet20 --epochs 5 --watermark-key 'secret_key'
    ```

- **5.2.2** **Evaluate and Verify**

  - Evaluate model performance.
  - Verify watermark detection using appropriate methods.

#### **5.3** **Compare All Methods**

- **Tasks:**
  - Create comparison tables for model accuracy, watermark robustness, and training time.
  - Analyze strengths and weaknesses of each method.
  - Discuss practical implications.

---

### **6. Analyze and Report Results**

- **6.1** **Compile Data**

  - Organize all recorded metrics and observations.
  - Use spreadsheets or data analysis tools.

- **6.2** **Perform Statistical Analysis**

  - Calculate mean, standard deviation, and other relevant statistics.
  - Conduct hypothesis testing if applicable.

- **6.3** **Create Visualizations**

  - Plot graphs showing:
    - Accuracy vs. `k`
    - Watermark detection rate vs. `k`
    - Comparison of different watermarking methods

- **6.4** **Draft Report**

  - **Sections to Include:**
    - Introduction and Objectives
    - Methodology
    - Results
    - Analysis and Discussion
    - Conclusion
  - **Discuss:**
    - Trade-offs between watermark robustness and model performance.
    - Optimal `k` value balancing accuracy and security.
    - Recommendations for practitioners.

---

### **7. Prepare for Advisor Review**

- **7.1** **Organize Code and Results**

  - Ensure code is well-documented.
  - Comment scripts for clarity.
  - Prepare a README for each experiment if necessary.

- **7.2** **Update Repository**

  - Push all changes to GitHub.
  - Ensure the repository is private or has appropriate access settings.

- **7.3** **Share Findings**

  - Send the report and link to the repository to your advisor.
  - Highlight key findings and invite feedback.

---

### **Additional Notes**

- **Consistency:** Keep hyperparameters consistent across experiments unless intentionally varying them.

- **Documentation:** Maintain a lab notebook or digital document recording all experiments, commands run, and any issues encountered.

- **Backup:** Regularly backup your data and models.

---

### **Sample Commands and How to Run Them**

**Training Without Watermarking (Baseline):**

```bash
python PoL/train.py --save-freq 100 --dataset CIFAR10 --model resnet20 --epochs 5
```

- **Explanation:** Trains a ResNet-20 model on CIFAR-10 for 5 epochs, saving checkpoints every 100 steps.

**Training With Feature-Based Watermarking:**

Replace `[K_VALUE]` with desired `k` value (e.g., 1000).

```bash
python PoL/train_with_watermark.py --save-freq 100 --dataset CIFAR10 --model resnet20 --epochs 5 --lambda-wm 0.01 --k [K_VALUE] --watermark-key 'secret_key'
```

- **Explanation:** Trains the model with feature-based watermarking.

**Verification of Proof-of-Learning and Watermark Presence:**

```bash
python PoL/verify.py --model-dir proof/CIFAR10_Batch100_k[K_VALUE] --dataset CIFAR10 --model resnet20 --epochs 5 --save-freq 100 --batch-size 128 --lr 0.1 --lambda-wm 0.01 --k [K_VALUE] --watermark-key 'secret_key' --dist 1 2 inf cos --delta 10000 100 1 0.1 --watermark-path model_with_watermark_k[K_VALUE].pth
```

- **Explanation:** Verifies the Proof-of-Learning and checks for the presence of the watermark in the specified model.

**Fine-Tuning the Watermarked Model:**

```bash
python PoL/fine_tune.py --model-path model_with_watermark_k[K_VALUE].pth --dataset CIFAR10 --model resnet20 --epochs 2
```

- **Explanation:** Fine-tunes the watermarked model without the watermark loss.

**Verification After Fine-Tuning:**

```bash
python PoL/verify.py --model-dir fine_tuned_model_k[K_VALUE] --dataset CIFAR10 --model resnet20 --epochs 5 --save-freq 100 --batch-size 128 --lr 0.1 --lambda-wm 0 --k [K_VALUE] --watermark-key 'secret_key' --dist 1 2 inf cos --delta 10000 100 1 0.1 --watermark-path fine_tuned_model_k[K_VALUE].pth
```

- **Explanation:** Checks if the watermark is still detectable after fine-tuning.

---

### **Questions for Clarification**

- **Implementation Status:**

  - Do you already have scripts for parameter perturbation-based and non-intrusive watermarking (`train_with_param_watermark.py` and `train_with_nonintrusive_watermark.py`), or do these need to be developed?

- **Attack Simulations:**

  - Do you require assistance in implementing scripts for pruning and other attack simulations?

- **Datasets and Models:**

  - Are there other datasets (e.g., CIFAR-100, ImageNet subset) or models (e.g., ResNet-50) you plan to include?

- **Performance Metrics:**

  - Would you like to include additional metrics like training time, model size, or memory usage?

- **Additional Experiments:**

  - Are there any specific scenarios or variations you wish to test (e.g., different `lambda-wm` values, randomizing watermark embedding intervals)?

---