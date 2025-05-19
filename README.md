# Shared Manifold for Internal Clustering Evaluation with Deep Learning

Deep clustering integrates deep neural networks into clustering pipelines, jointly learning embeddings and cluster assignments. However, evaluating and comparing the performance of these models is challenging, especially since results are typically assessed in different learned embedding spaces. This lack of a common evaluation ground introduces inconsistencies, as standard clustering metrics assume a shared feature space. To address this, we propose a framework that projects different embedding spaces into a **shared low-dimensional manifold**, enabling consistent internal evaluation. Our method preserves the similarity structure across models and improves the reliability and ranking consistency of evaluation metrics for deep clustering.

---
### Dependencies

This project requires the following Python packages:

* `numpy`
* `scipy`
* `scikit-learn==1.5` (version 1.5 is strictly required)

---

### Evaluation

1. Download the output files from the deep clustering algorithms listed in the paper:
   [Google Drive Folder](https://drive.google.com/drive/folders/1HBmYO_BThSW9ysfCPKPRuwm4kg1bKzSM?usp=drive_link)

2. Save them into a local directory.

3. Run the evaluation script `towards.py` to implement the experiments and obtain the results in the paper.


To evaluate a specific dataset or task:

```bash
python towards.py --task <task> --dataset <dataset>
```

#### Arguments

* `--task`: Specifies the task. Options:

  * `jule`, `julenum` (for JULE tasks)
  * `DEPICT`, `DEPICTnum` (for DEPICT tasks)
* `--dataset`: Name of the dataset to evaluate

---

### Output

The script outputs Kendall Tau and Spearman correlation scores between internal evaluation metrics computed on the learned shared manifold and external benchmarks such as NMI and clustering accuracy.