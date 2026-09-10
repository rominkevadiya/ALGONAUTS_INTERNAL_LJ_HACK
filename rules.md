# Project Rules

## 1. Development
- All substantive work must be committed between 10–15 September.
- Keep commits meaningful and related to actual development.
- Do not copy public notebooks or repositories wholesale.
- Open-source libraries, pretrained models, and public datasets are allowed.
- Properly credit third-party code, models, and datasets used.

## 2. Git & Collaboration
- Every member should work on their own branch.
- Do not directly push development work to the main branch.
- Create a Pull Request when your work is ready.
- Only the admin/maintainer should merge Pull Requests after review.
- Keep the repository clean, organized, and reproducible.

## 3. Data & ML
- Do not use the held-out test set for training.
- Maintain an honest train/validation/test split.
- Any additional public dataset must be properly cited.
- Avoid data leakage.
- Report results honestly, including limitations and failure cases.

## 4. Evaluation
- The core system must classify images as Real or AI-generated.
- Report ROC-AUC, Macro-F1, and a confusion matrix.
- Report performance on the unseen-generator split.
- Report accuracy and false-positive rate at the chosen threshold.
- Do not manipulate or replace the official evaluation data.

## 5. Responsible AI
- Present predictions as likelihoods, not absolute accusations.
- Do not target, identify, or profile real individuals.
- Do not make political or real-world event claims.
- Explanations must be grounded in actual visual evidence.
- Clearly communicate uncertainty and limitations.

## 6. Reproducibility
- Provide clear setup and execution instructions.
- Ensure the core system can be reproduced from the repository.
- Keep dependencies and required configurations documented.
- The final repository should allow judges to run and verify the system.

## 7. Documentation
- Keep the README updated with the implemented modules, datasets, approach, metrics, limitations, and demo information.
- Maintain an originality declaration for third-party code or references.
