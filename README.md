# Awesome-Uplift-Model
How to Apply Causal ML to Real Scene Modeling？How to learn Causal ML？

## Basic Theory

### Book Reading

- [The Book of Why](https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X/) by **Judea Pearl**, Dana Mackenzie
- [Causal Inference Book (What If)](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) by Miguel Hernán, **James Robins** **FREE download**
- [Causal Inference in Statistics: A Primer](https://www.amazon.com/Causal-Inference-Statistics-Judea-Pearl/dp/1119186846/) by Judea Pearl, Madelyn Glymour, Nicholas P. Jewell
- [Elements of Causal Inference: Foundations and Learning Algorithms](https://mitpress.mit.edu/books/elements-causal-inference) by Jonas Peters, Dominik Janzing and Bernhard Schölkopf- **FREE download**
- [Counterfactuals and Causal Inference: Methods and Principles for Social Research](https://www.amazon.com/Counterfactuals-Causal-Inference-Principles-Analytical/dp/1107694167) by Stephen L. Morgan, Christopher Winship
- [Causal Inference Book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) by Hernán MA, Robins JM **FREE download**
- [Causality: Models, Reasoning and Inference](https://www.amazon.com/Causality-Reasoning-Inference-Judea-Pearl/) by Judea Pearl
- [Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction](https://www.amazon.com/Causal-Inference-Statistics-Biomedical-Sciences/dp/0521885884/) by Guido W. Imbens and Donald B. Rubin
- [Causal Inference: The Mixtape](https://www.scunning.com/mixtape.html) by Scott Cunningham **FREE download**
- [Causal Inference for Data Science](https://www.manning.com/books/causal-inference-for-data-science) by Aleix Ruiz de Villa

The most commonly used models for causal inference are Rubin Causal Model (RCM; Rubin 1978) and Causal Diagram (Pearl 1995). Pearl (2000) introduced the equivalence of these two models, but in terms of application, RCM is more accurate, while Causal Diagram is more intuitive, which is highly praised by computer experts.

> Donald Bruce Rubin (born December 22, 1943) is an Emeritus Professor of Statistics at [Harvard University](https://en.wikipedia.org/wiki/Harvard_University). He is most well known for the [Rubin causal model](https://en.wikipedia.org/wiki/Rubin_causal_model), a set of methods designed for [causal](https://en.wikipedia.org/wiki/Causal) [inference](https://en.wikipedia.org/wiki/Inference) with [observational data](https://en.wikipedia.org/wiki/Observational_data), and for his methods for dealing with [missing data](https://en.wikipedia.org/wiki/Missing_data).
>
> Judea Pearl (born September 4, 1936) is an Israeli-American [computer scientist](https://en.wikipedia.org/wiki/Computer_scientist) and [philosopher](https://en.wikipedia.org/wiki/Philosopher), best known for championing the probabilistic approach to [artificial intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence) and the development of [Bayesian networks](https://en.wikipedia.org/wiki/Bayesian_networks) (see the article on [belief propagation](https://en.wikipedia.org/wiki/Belief_propagation)).

### More Details

> #### Pearl's Structural Causal Model
>
> - **The book of why: The new science of cause and effect**
>      *by Judea Pearl and Dana Mackenzie, 2018. [**Get Book**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=bAipNH8AAAAJ&citation_for_view=bAipNH8AAAAJ:EsrhoZGmrkoC)*
>   **[Must Read]** An amazing beginner's guide to graph-based causality models.
> - **Causal inference in statistics: A primer**
>      *by Madelyn Glymour, Judea Pearl, Nicholas P Jewell, 2016. [**Get Book**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=bAipNH8AAAAJ&cstart=20&pagesize=80&citation_for_view=bAipNH8AAAAJ:35N4QoGY0k4C)*
>   **[Must Read]** The essense of causal graph, adjustment, and counterfactuals in FOUR easy-to-follow chapters.
> - **Causality: Models, Reasoning, and Inference**
>      *by Judea Pearl, 2009. [**Get Book**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=bAipNH8AAAAJ&citation_for_view=bAipNH8AAAAJ:u-x6o8ySG0sC)*
>   **[Suggested]** A formal and comprehensive discussion of every corner of Pearl's causality.
>
> #### Rubin's Potential Outcome Model
>
> - **Causal inference in statistics, social, and biomedical sciences**
>      *Guido W Imbens, Donald B Rubin, 2015. [**Get Book**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=5q4fhUoAAAAJ&citation_for_view=5q4fhUoAAAAJ:QeguYG95ZbAC)*
>   **[Must Read]** A formal and comprehensive discussion of Rubin's potential outcome framework.
>
> #### A Mixure of Both Frameworks
>
> - **Causal Inference for The Brave and True**
>      *Matheus Facure, 2021. [**Get Book**](https://matheusfacure.github.io/python-causality-handbook)*
>   **[Must Read]** A new book that describes causality in an amazing mixture of Pearl's and Rubin's frameworks.
>
> #### Disputes between Pearl and Rubin
>
> Not necessarily books. Posts and papers are included.
>
> ##### From Andrew Gelman (Student of Rubin, now Prof. at Columbia U.)
>
> - **Resolving disputes between J. Pearl and D. Rubin on causal inference** [[**Go to post**\]](https://statmodeling.stat.columbia.edu/2009/07/05/disputes_about/)
>   **[Must Read]** The post from Prof. Gelman shows the disputes from Rubin's perspective. It helps understand why Pearl's framework faces great challenges in the statistic community while being so successful in machine learning and social computing.
> - **“The Book of Why” by Pearl and Mackenzie** [[**Go to post**\]](https://statmodeling.stat.columbia.edu/2019/01/08/book-pearl-mackenzie/)
>   **[Must Read]** Critics from Rubin's causal perspective to the famous guiding book for causality: The book of why.
>
> ##### From Judea Pearl (Prof. at UCLA)
>
> - **Chapter 8, The Book of Why?** [[**Get book**\]](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=bAipNH8AAAAJ&citation_for_view=bAipNH8AAAAJ:EsrhoZGmrkoC)
>   **[Must Read]** Pearl's overall discussion of the short comings of Rubin's potential outcome framework.
> - **Can causal inference be done in statistical vocabulary?** [[**Go to post**\]](http://causality.cs.ucla.edu/blog/index.php/2019/01/09/can-causal-inference-be-done-in-statistical-vocabulary/)
>   **[Must Read]** Pearl's initial reponse to Gelman's critics on The book of why.
> - **More on Gelman’s views of causal inference** [[**Go to post**\]](http://causality.cs.ucla.edu/blog/index.php/2019/01/15/more-on-gelmans-views-of-causal-inference/)
>   **[Must Read]** Pearl's next reponse to Gelman's critics on The book of why.



### Code Examples

- 《[Causal Inference The Mixtape](https://mixtape.scunning.com/)》| [Code Example](./Example/Causal_Inference/)
- 《[The Effect](https://theeffectbook.net/index.html)》| [Code Example](./Example/The_Effect/)



## Courses

- [Introduction to Causal Inference (Fall2020)](https://www.bradyneal.com/causal-inference-course) (Free)
- [A Crash Course in Causality: Inferring Causal Effects from Observational Data](https://www.coursera.org/learn/crash-course-in-causality) (Free)
- [Causal Inference with R - Introduction](https://www.datacamp.com/community/open-courses/causal-inference-with-r-introduction) (Free)
- [Causal ML Mini Course](https://altdeep.ai/p/causal-ml-minicourse) (Free)
- [Lectures on Causality: 4 Parts](https://www.youtube.com/watch?v=zvrcyqcN9Wo) by Jonas Peters
- [Towards Causal Reinforcement Learning (CRL) - ICML'20 - Part I](https://slideslive.com/38930490/towards-causal-reinforcement-learning-crl-part-i?ref=speaker-22075-latest) By Elias Bareinboim
- [Towards Causal Reinforcement Learning (CRL) - ICML'20 - Part II](https://slideslive.com/38930491/towards-causal-reinforcement-learning-part-ii?ref=speaker-22075-latest) By Elias Bareinboim
- [On the Causal Foundations of AI](https://www.youtube.com/watch?v=fNuMHDrh6AY&t=31s) By Elias Bareinboim
- [Judea Pearl: Causal Reasoning, Counterfactuals, and the Path to AGI | Lex Fridman Podcast #56](https://www.youtube.com/watch?v=pEBI0vF45ic) By Judea Pearl and Lex Fridman
- [NeurIPS 2018 Workshop on Causal Learning](https://www.youtube.com/playlist?list=PLJscN9YDD1bu1dCKuXSV1qYmicx3g9t7A)
- [Causal Inference Bootcamp](https://mattmasten.github.io/bootcamp/) by Matt Masten



## Tools

### Probabilistic programming framework

- [pyro](http://pyro.ai/)
- [pymc3](http://docs.pymc.io/)
- [pgmpy](https://github.com/pgmpy/pgmpy)
- [pomegranate](https://github.com/jmschrei/pomegranate)

### Causal Structure Learning

- [TETRAD](https://github.com/cmu-phil/tetrad)
- [CausalDiscoveryToolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox)
- [gCastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle)
- [tigramite](https://github.com/jakobrunge/tigramite)

### Causal Inference

- [Ananke](https://ananke.readthedocs.io/en/latest/)
- [EconML](https://github.com/microsoft/EconML)
- [dowhy](https://github.com/microsoft/dowhy)
- [**causalml**](https://github.com/uber/causalml)
- [WhyNot](https://whynot.readthedocs.io/en/latest/)
- [CausalImpact](https://github.com/google/CausalImpact)
- [Causal-Curve](https://github.com/ronikobrosly/causal-curve)
- [grf](https://github.com/grf-labs/grf)
- [dosearch](https://cran.r-project.org/web/packages/dosearch/index.html)
- [causalnex](https://github.com/quantumblacklabs/causalnex)



## Datasets and Benchmark

### Causal Inference

- MIMIC II/III Data：ICU数据
  - [Data1](https://archive.physionet.org/mimic2/)
  - [Data2](https://physionet.org/content/mimiciii/1.4/)
- [Advertisement Data](https://research.google/pubs/pub41854/)：广告数据
- [Geo experiment data](https://research.google/pubs/pub45950/)：地理数据
- [Economic data for Spanish regions](https://www.aeaweb.org/articles?id=10.1257/000282803321455188)：没有Ground Truth
- [California’s Tobacco Control Program](https://economics.mit.edu/files/11859)：
- [Air Quality Data](https://www.aeaweb.org/articles?id=10.1257/aer.101.6.2687)：
- [Monetary Policy Data](https://www.tandfonline.com/doi/abs/10.1080/01621459.2018.1491403?journalCode=uasa20)：
- [JustCause](https://justcause.readthedocs.io/en/latest/)：Benchmark

### Causal Discovery

- [Causal Inference for Time series Analysis: Problems, Methods and Evaluation](https://arxiv.org/abs/2102.05829)
- [Causeme](https://causeme.uv.es/)：Benchmark
- Real Dataset：
  - US Manufacturing Growth Data
  - Diabetes Dataset
  - Temperature Ozone Data
  - OHDNOAA Dataset 
  - Neural activity Dataset
  - Human Motion Capture
  - Traffic Prediction Dataset
  - Stock Indices Data
- Composite Dataset：
  - Confounding/ Common-cause Models
  - Non-Linear Models
  - Dynamic Models
  -  Chaotic Models



## Other Awesome List

- [awesome-causality-algorithms](https://github.com/rguo12/awesome-causality-algorithms)
- [awesome-causality-data](https://github.com/rguo12/awesome-causality-data)
- [awesome-causality](https://github.com/napsternxg/awesome-causality)
- [Awesome-Causality-in-CV](https://github.com/Wangt-CN/Awesome-Causality-in-CV)
- [Awesome-Neural-Logic](https://github.com/FLHonker/Awesome-Neural-Logic)
- [Awesome-Causal-Inference](https://github.com/matthewvowels1/Awesome-Causal-Inference)



## How to Apply Causal ML to Real Scene Modeling？

- [Basic Conception](./Basic_Theory.md)
- [ATE Method](./ATE_Method.md)
- [Basic of Uplift](./Basic_Uplift_Model.md)
- [Uplift Modeling](./Uplift_Modeling.md)
- [Debias](./Debias.md)
- [Causal ML Framework](./Causal_ML_Framework.md)



## Contact

© [JackHCC](https://github.com/JackHCC) 2022



