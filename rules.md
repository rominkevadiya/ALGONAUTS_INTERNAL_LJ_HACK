# Project Rules

## 1. Development

* All substantive project work must be committed between **10–15 September**.
* Keep commits meaningful and related to actual development.
* Do not copy public notebooks or repositories wholesale.
* Open-source libraries, pretrained models, and public datasets are allowed.
* Properly credit all third-party code, models, and datasets used.

## 2. Git & Collaboration

* Every member must work on their **own branch**.
* Do not directly push development work to the `main` branch.
* Create a **Pull Request (PR)** when your work is ready.
* Only the **admin/maintainer** can approve and merge Pull Requests after review.
* Keep the repository clean, organized, and reproducible.

## 3. Data & ML

* Do not use the **held-out test set** for training.
* Maintain an honest **train/validation/test split**.
* Properly cite any additional public datasets.
* Avoid data leakage.
* Report results honestly, including limitations and failure cases.

## 4. Evaluation

* The core system must classify images as **Real** or **AI-generated**.
* Report:

  * **ROC-AUC**
  * **Macro-F1**
  * **Confusion Matrix**
  * Performance on the **unseen-generator split**
  * **Accuracy**
  * **False-positive rate (FPR)** at the chosen threshold
* Do not manipulate, replace, or selectively modify the official evaluation data.

## 5. Responsible AI

* Present predictions as **likelihoods**, not absolute accusations.
* Do not target, identify, or profile real individuals.
* Do not make political or real-world event claims.
* Explanations must be grounded in actual visual evidence.
* Clearly communicate uncertainty and system limitations.

## 6. Reproducibility

* Provide clear setup and execution instructions.
* Ensure the core system can be reproduced from the repository.
* Keep dependencies and required configurations documented.
* The final repository must allow judges to run and verify the system.

## 7. Documentation

* Keep the `README.md` updated with:

  * Implemented modules
  * Datasets
  * Approach
  * Evaluation metrics
  * Limitations
  * Demo information
* Maintain an **originality declaration** covering third-party code, models, datasets, and references used.
