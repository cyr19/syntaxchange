# syntaxchange

This repo contains the code+data for my master thesis on the topic of "syntactic language change in German and English" at TU Darmstadt as well as the resulting paper [Syntactic Language Change in English and German: Metrics, Parsers, and Convergences](https://arxiv.org/abs/2402.11549).

> **Abstract**:
> Syntactic language change has gained increasing attention in recent years. Previous computational work based on dependency relations has focused on diachronic trends in dependency distance, which measures the linear distance between dependent words, using dependency trees automatically predicted by a dependency parser (mostly the Stanford CoreNLP parser). In this work, we introduce a set of 15 syntax metrics that extend the analysis beyond linear distance by incorporating both linear and tree graph properties of dependency trees, such as tree height and degree. Besides, we propose a multi-parser approach to reduce the impact of using specific parsers, thereby increasing the robustness of the detected language changes. Through a cross-lingual investigation of English and German in parliamentary debates for the last 160 years, using 6 different parsers (CoreNLP and 5 newer alternatives), we demonstrate that: (1) Relying on one single parser can be problematic, as the agreement on predicted trends can be low across parsers. (2) Our set of metrics can capture subtle patterns of syntactic changes. Our analysis shows that syntactic change over the time period inspected is largely similar between English and German, with only 2.2% of cases yielding opposite trends in these metrics. (3) We also show that changes in syntactic metrics seem to be more frequent at the tails of sentence length distributions and often move in opposite directions for short and long sentences. To our best knowledge, ours is the most comprehensive computational analysis of syntactic language change using modern NLP technology in recent corpora of English and German.

## Overview

Check `parser.yaml` for the environment configuration used in this work. You don't need all packages to run the code here, especially if you are not using the parsers.

- [`code/data_process`](code/data_process): Contains code for data processing and validation results.

- [`code/parsers`](code/parsers): Originally contained the parsing code; however, we have moved it to a separate repository, [**LCPar**](https://github.com/cyr19/LCPar), for linguists who may want to use the parsers for their own research.

- [`code/analysis`](code/analysis): Includes code for analyzing language changes, such as calculating syntax metrics.

- [`data/`](data/): Contains sampled sentences from the original political corpora, parsing results, and other related files. 

## Citation
If you use code/data in this repo, please cite us!
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

