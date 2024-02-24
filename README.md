# syntaxchange

Will be updated to correspond with the latest version of our paper [Syntactic Language Change in English and German: Metrics, Parsers, and Convergences](https://arxiv.org/abs/2402.11549) soon!

> **Abstract**:
> Many studies have shown that human languages tend to optimize for lower complexity and increased communication efficiency. Syntactic dependency distance, which measures the linear distance between dependent words, is often considered a key indicator of language processing difficulty and working memory load. The current paper looks at diachronic trends in syntactic language change in both English and German, using corpora of parliamentary debates from the last c. 160 years. We base our observations on five dependency parsers, including the widely used Stanford CoreNLP as well as 4 newer alternatives. Our analysis of syntactic language change goes beyond linear dependency distance and explores 15 metrics relevant to dependency distance minimization (DDM) and/or based on tree graph properties, such as the tree height and degree variance. Even though we have evidence that recent parsers trained on modern treebanks are not heavily affected by data 'noise' such as spelling changes and OCR errors in our historic data, we find that results of syntactic language change are sensitive to the parsers involved, which is a caution against using a single parser for evaluating syntactic language change as done in previous work. We also show that syntactic language change over the time period investigated is largely similar between English and German across the different metrics explored: only 4% of cases we examine yield opposite conclusions regarding upwards and downtrends of syntactic metrics across German and English. We also show that changes in syntactic measures seem to be more frequent at the tails of sentence length distributions. To our best knowledge, ours is the most comprehensive analysis of syntactic language using modern NLP technology in recent corpora of English and German.

```
@misc{chen2024syntactic,
      title={Syntactic Language Change in English and German: Metrics, Parsers, and Convergences}, 
      author={Yanran Chen and Wei Zhao and Anne Breitbarth and Manuel Stoeckel and Alexander Mehler and Steffen Eger},
      year={2024},
      eprint={2402.11549},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Code and data for my master thesis on the topic of "syntactic language change in German and English" at TU Darmstadt.

The code for data processing and the validation results are located in [code/data_process](code/data_process).

The code for parsing is located in [code/parsers](code/parsers).

The code and results for analyzing language change are located in [code/analysis](code/analysis).

Needs to be cleaned.
