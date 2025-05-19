# Shared Manifold for Internal Clustering Evaluation with Deep Learning

Deep clustering integrates deep neural networks into clustering pipelines, jointly learning embeddings and cluster assignments. Significant challenges remain in evaluating and comparing the performance of different algorithms. The curse of dimensionality makes it impractical to evaluate clustering results directly in the original high-dimensional input space. As a result, clustering outcomes are commonly compared within the respective learned lower-dimensional embedding spaces of different models. However, this practice introduces inconsistencies, as existing clustering evaluation metrics are typically designed for comparisons within a single, shared feature space. These discrepancies complicate model validation and the evaluation of results in the context of deep clustering. To address these challenges, we propose an evaluation framework that enables the comparison of clustering results within a common low-dimensional feature space. Specifically, we introduce an approach to learn a shared manifold that preserves the underlying similarity structure across different embedding spaces while enabling effective internal evaluation. Through extensive experiments comparing clustering results across various models and training runs, our findings reveal that conducting evaluations on the shared manifold significantly improves the ranking consistency of evaluation measures, thereby enhancing the reliability of deep clustering performance assessments.


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