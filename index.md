## Abstract
Machine learning on graph data has become a common area of interest across academia and industry. However, due to the size of real-world industry graphs (hundreds of millions of vertices and billions of edges) and the special architecture of graph neural networks, it is still a challenge for practitioners and researchers to perform machine learning tasks on large-scale graph data. It typically takes a powerful and expensive GPU machine to train a graph neural network on a million-vertex scale graph, let alone doing deep learning on real enterprise graphs. In this tutorial, we will cover how to develop and run performant graph algorithms and graph neural network models with TigerGraph [3], a massively parallel platform for graph analytics, and its Machine Learning Workbench with PyTorch Geometric [4] and DGL [8] support. Using an NFT transaction dataset [6], we will first investigate transactions using graph algorithms by themselves as methods of graph traversing, clustering, classification, and determining similarities between data. Secondly, we will show how to use those graph-derived features such as PageRank and embeddings to empower traditional machine learning models. Finally, we will demonstrate how to train common graph neural networks with TigerGraph and how to implement novel graph neural network models. Participants will use the TigerGraph ML Workbench Cloud to perform graph feature engineering and train their machine learning algorithms during the session.

## Tutorial Contents
1) [Getting Started](getting_started)

2) [Graph Data Science Notebooks](notebooks)

3) [Next Steps](next_steps)

## Authors

### Victor Lee, Ph.D.
**VP of Machine Learning and AI, TigerGraph**

Victor wrote his Ph.D. dissertation on graph-based similarity and ranking. Dr. Lee has co-authored book chapters on decision trees and dense subgraph discovery. Teaching and training have also been central to his career journey, with activities ranging from developing training materials for chip design to writing the first version of TigerGraph’s technical documentation, from teaching 12 years as a full-time or part-time university instructor, to presenting numerous webinars and in-person workshops.

### Parker Erickson
**Machine Learning Engineer, TigerGraph**

Parker received his M.S. of Computer Science from the University of Minnesota with a focus on machine learning in 2022. He is also the founding developer of pyTigerGraph, a Python package for interacting with TigerGraph databases. Formerly an intern at UnitedHealth Group, Parker worked on graph machine learning solutions for fraud detection, member similarity, training course recommendation, pharmacy formulary rules, and call center analytics.

### Feng Shi, Ph.D.
**Sr ML Architect and Manager, TigerGraph**

Bill received his PhD in Applied Math with a research focus on complex networks. His work has been published in top academic journals and received coverage from widely circulated journals and news outlets, including Nature, The Guardian, The Los Angeles Times, and so on. Prior to joining TigerGraph, he led the Books Knowledge Graph initiative and graph machine learning research at Amazon and led the Social Network Analysis at Carolina program at UNC Chapel Hill. He has served as session organizer and chair for top academic conferences including NetSci 2017 and 2019, and Joint Mathematics Meetings 2018 and 2019. He also serves as a guest editor for the journals of Frontiers. He has taught various network analysis courses including one at datamatters.org and authored dozens of tutorials on data analysis for SAGE Research Methods.

### Jiliang Tang, Ph.D.
**Professor, Michigan State University**

He was an associate professor from 2021 to 2022 and an assistant professor from 2016 to 2021 in the same department His research interests include data mining, machine learning and their applications in social media and education. He was the recipient of 2022 SDM IBM Early Career Data Mining Research Award, 2021 ICDM Tao Li Award, 2020 SIGKDD Rising Star Award, 2019 NSF Career Award, and 8 best paper awards (or runner-ups) including WSDM2018 and KDD2016. His dissertation won the 2015 KDD Best Dissertation runner up and Dean’s Dissertation Award. He serves as top data science conference organizers (e.g., KDD, SIGIR, WSDM, and SDM) and journal editors (e.g., TKDD and TKDE). He has published his research in highly ranked journals and top conference proceedings, which received more than 21,000 citations with h-index 69 and extensive media coverage

## References
[1] Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. 2008. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment 2008, 10 (Oct 2008), P10008. https://doi.org/10.1088/1742-5468/2008/10/p10008

[2] Tianqi Chen and Carlos Guestrin.2016.XGBoost:AScalableTreeBoostingSystem. CoRR abs/1603.02754 (2016). arXiv:1603.02754 http://arxiv.org/abs/1603.02754

[3] Alin Deutsch, Yu Xu, Mingxi Wu, and Victor E. Lee. 2019. TigerGraph: A Native MPP Graph Database. CoRR abs/1901.08248 (2019). arXiv:1901.08248 http://arxiv.org/abs/1901.08248

[4] Matthias Fey and Jan Eric Lenssen. 2019. Fast Graph Representation Learning with PyTorch Geometric. CoRR abs/1903.02428 (2019). arXiv:1903.02428 http://arxiv.org/abs/1903.02428

[5] William L. Hamilton, Rex Ying, and Jure Leskovec. 2017. Inductive Representation Learning on Large Graphs. CoRR abs/1706.02216 (2017). arXiv:1706.02216 http://arxiv.org/abs/1706.02216

[6] Matthieu Nadini, Laura Alessandretti, Flavio Di Giacinto, Mauro Martino, Luca Maria Aiello, and Andrea Baronchelli. 2021. Mapping the NFT revolution: market trends, trade networks, and visual features. Scientific Reports 11, 1 (Oct 2021). https://doi.org/10.1038/s41598-021-00053-8

[7] Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. 1999. The PageRank Citation Ranking: Bringing Order to the Web. Technical Report 1999-66. Stanford InfoLab. http://ilpubs.stanford.edu:8090/422/ Previous number = SIDL-WP-1999-0120.

[8] Minjie Wang, Lingfan Yu, Da Zheng, Quan Gan, Yu Gai, Zihao Ye, Mufei Li, Jinjing Zhou, Qi Huang, Chao Ma, Ziyue Huang, Qipeng Guo, Hao Zhang, Haibin Lin, Junbo Zhao, Jinyang Li, Alexander J. Smola, and Zheng Zhang. 2019. Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs. CoRR abs/1909.01315 (2019). arXiv:1909.01315 http://arxiv.org/abs/1909.01315