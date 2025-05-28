# Informative Embedding Ensemble: Towards a Consensus Space for Reliable Clustering Evaluation in Deep Learning

Deep clustering methods have seen extensive development in enhancing the scalability of traditional clustering techniques. By transforming high-dimensional data into a lower-dimensional latent feature space (also known as the embedding space) using deep neural networks, these methods make the clustering process more efficient and manageable. Most deep clustering approaches optimize a clustering objective based on the learned embedding space, addressing the challenges associated with high-dimensional data. Despite these advancements, accurately evaluating and validating the model performance remains a significant challenge, particularly due to the absence of labels. Proper evaluation is crucial for both model training and comparison, yet it remains an under-explored aspect of deep clustering research. 


---
### Dependencies

This project requires the following Python packages:

* `numpy`
* `scipy`
* `scikit-learn==1.5` (version 1.5 is strictly required)

---

### Evaluation

1. Download the evaluated output files from the deep clustering algorithms:
   [Google Drive Folder](https://drive.google.com/drive/folders/1HBmYO_BThSW9ysfCPKPRuwm4kg1bKzSM?usp=drive_link)

2. Save them into the same local directory with the scripts.

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

The script outputs Kendall Tau and Spearman correlation coefficients between internal evaluation metrics computed on the learned shared manifold and external benchmarks such as NMI and clustering accuracy.
