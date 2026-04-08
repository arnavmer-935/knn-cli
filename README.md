# KNN CLI Tool

Interactive command-line tool for K-Nearest Neighbors (KNN) classification and evaluation on real-world datasets.

---

## Features
- Interactive prompt-based CLI (no flags required)
- Supports classification and evaluation workflows
- Multiple distance metrics: Euclidean, Manhattan, Cosine
- Optional feature normalization (z-score, min-max)
- Built-in validation for all user inputs
- Optional data visualization (2D/3D plots)

---

## Installation

```bash
  pip install -e .
```

---

## Usage

```bash
  knn-cli
```

Follow the interactive prompts to:
- Load a dataset
- Choose k
- Select distance metric
- Configure normalization
- Run classification or evaluation

---

## Dataset Requirements
- CSV file with header row
- At least 2 columns
- Numeric feature columns

---

## Example Workflow
1. Provide dataset path
2. Select k value
3. Choose distance metric
4. Enable/disable normalization
5. Choose classification or evaluation
6. View results and optional plots

---

## Tech Stack
- Python
- Typer (CLI framework)
- Rich (terminal UI)
- Matplotlib (visualization)

---

## Notes
- Press `Ctrl+C` at any time to exit
- Input validation is enforced at every step