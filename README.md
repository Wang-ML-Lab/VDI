# Domain-Indexing Variational Bayes: Interpretable Domain Index for Domain Adaptation (VDI) (Updating)
This repo contains the code for our ICLR 2023 paper:<br>
**Domain-Indexing Variational Bayes: Interpretable Domain Index for Domain Adaptation**<br>
Zihao Xu*, Guang-Yuan Hao*, Hao He, Hao Wang<br>
*Eleventh International Conference on Learning Representations, 2023*<br>
[[Paper](https://arxiv.org/abs/2302.02561)]

## Outline for This README
* [Brief Introduction for VDI](#brief-introduction-for-vdi)
* [Sample Visualization of Inferred Domain Indices](#sample-visualization-of-inferred-domain-indices)
* [Domain Index Definition (Informal)](#domain-index-definition-informal)
* [Method Overview](#method-overview)
* [Quantitative Result](#quantitative-result)
* [More Visualization of Inferred Domain Indices](#more-visualization-of-inferred-domain-indices)
* [Installation](#installation)
* [Training/Visualizing Steps](#trainvisualize-steps)
* [Related Works](#also-check-our-relevant-work)
* [Reference](#reference)

## Brief Introduction for VDI
Previous studies have shown that leveraging domain index can significantly boost domain adaptation performance [[1](#paper_1),[2](#paper_2)]. However, such domain indices are not always available. VDI is the model that aims to address this challenge. To achieve this goal, we first formally define the "domain index" from the probabilistic perspective, and then **infers** domain indices from multi-domain data, with an **adversarial variational Bayesian framework**. These domain indice provide additional insight on domain relations and improve domain adaptation performance. Our theoretical analysis shows that VDI finds the optimal domain index at equilibrium.
<!-- Empirical results on both synthetic and real data verify that VDI can produce interpretable domain indices which enable us to achieve superior performance compared to state-of-the-art domain adaptation methods. -->

## Sample Visualization of Inferred Domain Indices
Below are inferred domain indices for $48$ domains in *TPT-48*. We color inferred domain indices according to ground-truth indices, latitude (**left**) and longitude (**right**). VDI's inferred indices are correlated with true indices, even though *VDI does not have access to true indices during training*.
<p align="center">
<img src="fig/visualize_tpt_48.jpg" alt="" data-canonical-src="fig/visualize_tpt_48.jpg" width="93%"/>
</p>
We could see that VDI's inferred domain indices are highly correlated with each domain's latitude and longitude. For example, Florida (FL) has the lowest latitude among all 48 states and is hence the left-most circle in the left figure.

## Domain Index Definition (Informal)
We require domain index:
* Be independent of the data's encoding (Domain-Invariant).
* Retains as much information on the data as possible.
* Maximizes adaptation performance.

## Method Overview
We propose a Hierarchical Bayes model for domain index inference, which is shown below.  **Left:** Probabilistic graphical model for VDI's generative model. **Right:** Probabilistic graphical model for the VDI's inference model. See our [paper](https://arxiv.org/abs/2302.02561) for detailed explanation.
<p align="center">
<img src="fig/graphic_model.jpg" alt="" data-canonical-src="fig/graphic_model.jpg" width="70%"/>
</p>
Our theortical analysis found that **maximizing our model's evidence lower bound** while **adversarially training** an additional discriminator is equivalent to **inferring the optimal domain indices** according to the [definition](#domain-index-definition-informal). This gives rise to our final network structure shown below.
<p align="center">
<img src="fig/network_structure.jpg" alt="" data-canonical-src="fig/network_structure.jpg" width="89%"/>
</p>

## Quantitative Result
#### Toy Dataset: Circle, DG-15 and DG-60
<p align="center">
<img src="fig/toy_quantitive_result.jpg" alt="" data-canonical-src="fig/toy_quantitive_result.jpg" width="80%"/>
</p>

#### TPT-48
<p align="center">
<img src="fig/tpt_48_quantitive_result.jpg" alt="" data-canonical-src="fig/tpt_48_quantitive_result.jpg" width="95%"/>
</p>

#### CompCars
<p align="center">
<img src="fig/compcar_quantitive_result.jpg" alt="" data-canonical-src="fig/compcar_quantitive_result.jpg" width="80%"/>
</p>

## More Visualization of Inferred Domain Indices
#### Circles
<p align="center">
<img src="fig/visualize_circle.jpg" alt="" data-canonical-src="fig/visualize_circle.jpg" width="65%"/>
</p>

#### CompCars
<p align="center">
<img src="fig/visualize_compcar_view.jpg" alt="" data-canonical-src="fig/visualize_compcar_view.jpg" width="70%"/>
</p>
<p align="center">
<img src="fig/visualize_compcar_YOM.jpg" alt="" data-canonical-src="fig/visualize_compcar_YOM.jpg" width="70%"/>
</p>

#### DG-15
<p align="center">
<img src="fig/dg15_graph.jpg" alt="" data-canonical-src="fig/dg15_graph.jpg" width="80%"/>
</p>

## Installation
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    pip install -r requirements.txt

## Training/Visualizing Steps
In the directory of each dataset, there are detailed steps on how to train VDI and how to visualize the inferred domain indices.

## Also Check Our Relevant Work
<span id="paper_1">[1] **Graph-Relational Domain Adaptation**<br></span>
Zihao Xu, Hao He, Guang-He Lee, Yuyang Wang, Hao Wang<br>
*Tenth International Conference on Learning Representations (ICLR), 2022*<br>
[[Paper](http://wanghao.in/paper/ICLR22_GRDA.pdf)] [[Talk](https://www.youtube.com/watch?v=oNM5hZGVv34)] [[Slides](http://wanghao.in/slides/GRDA_slides.pptx)]

<span id="paper_2">[2] **Continuously Indexed Domain Adaptation**<br></span>
Hao Wang*, Hao He*, Dina Katabi<br>
*Thirty-Seventh International Conference on Machine Learning (ICML), 2020*<br>
[[Paper](http://wanghao.in/paper/ICML20_CIDA.pdf)] [[Code](https://github.com/hehaodele/CIDA)] [[Talk](https://www.youtube.com/watch?v=KtZPSCD-WhQ)] [[Blog](http://wanghao.in/CIDA-Blog/CIDA.html)] [[Slides](http://wanghao.in/slides/CIDA_slides.pptx)]

## Reference
[Domain-Indexing Variational Bayes: Interpretable Domain Index for Domain Adaptation](https://arxiv.org/abs/2302.02561)
```bib
@inproceedings{VDI,
  title={Domain-Indexing Variational Bayes: Interpretable Domain Index for Domain Adaptation},
  author={Xu, Zihao and Hao Guang-Yuan and He, Hao and Wang, Hao},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
