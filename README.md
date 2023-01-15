# Integrated Master's Project: Dataset Distillation for EuroSAT

This repos contains the code for the Integrative Master's Project (IMP) of the Master's in Computer Science at Hochschule St. Gallen conducted by Daniel Leal and Julius Lautz. The IMP was conducted under the supervision of Prof. Dr. Michael Mommert and Linus Scheibenreif at the Chair of Artifical Intelligence and Machine Learning (AI:ML).

A majority of the code was taken from the [paper](https://arxiv.org/abs/2203.11932) "Dataset Distillation by Matching Training Trajectories" from Cazenavette et al. ([code](https://github.com/GeorgeCazenavette/mtt-distillation) available here) and applied to the [EuroSAT](https://github.com/phelber/EuroSAT) dataset.


### Getting Started

First, download our repo:
```bash
git clone https://github.com/julius-lautz/imp_dataset_distillation.git
cd imp_dataset_distillation
```

Then, you can install all dependencies with the following command:
```bash
pip -r requirements.txt
```

### Generating Expert Trajectories
Before doing any distillation, you'll need to generate some expert trajectories using ```expert_trajectories.py```

The following command will train 100 ConvNet models on EuroSAT with ZCA whitening for 50 epochs each:
```bash
python expert_trajectories.py --dataset=EuroSAT --model=ConvNet --train_epochs=50 --num_experts=100 --zca --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset}
```
The other models available are an MLP and a ResNet18 (set model to either ```--model="MLP"``` or ```--model="ResNet"```).

### Distillation by Matching Training Trajectories
The following command will then use the expert trajectories we just generated to distill EuroSAT down to just 1 image per class:
```bash
python distillation.py --dataset=EuroSAT --ipc=1 --syn_steps=20 --expert_epochs=3 --max_start_epoch=5 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset}
```

<img src='docs/animation.gif' width=600>

Please find a full list of hyper-parameters below:

![image](https://user-images.githubusercontent.com/18726777/184226412-7bd0d577-225b-487c-8c9c-23f6462ca7d0.png)



## Related Work
<ol>
<li>
    Tongzhou Wang et al. <a href="https://ssnl.github.io/dataset_distillation/">"Dataset Distillation"</a>, in arXiv preprint 2018
</li>
<li>
    Bo Zhao et al. <a href="https://arxiv.org/abs/2006.05929">"Dataset Condensation with Gradient Matching"</a>, in ICLR 2020
</li>
<li>
    Bo Zhao and Hakan Bilen. <a href="https://arxiv.org/abs/2102.08259">"Dataset Condensation with Differentiable Siamese Augmentation"</a>, in ICML 2021
</li>
<li>
    Timothy Nguyen et al. <a href="https://arxiv.org/abs/2011.00050">"Dataset Meta-Learning from Kernel Ridge-Regression"</a>, in ICLR 2021
</li>
<li>
    Timothy Nguyen et al. <a href="https://arxiv.org/abs/2107.13034">"Dataset Distillation with Infinitely Wide Convolutional Networks"</a>, in NeurIPS 2021
</li>
<li>
    Bo Zhao and Hakan Bilen. <a href="https://arxiv.org/abs/2110.04181">"Dataset Condensation with Distribution Matching"</a>, in arXiv preprint 2021
</li>
<li>
    Kai Wang et al. <a href="https://arxiv.org/abs/2203.01531">"CAFE: Learning to Condense Dataset by Aligning Features"</a>, in CVPR 2022
</li>
</ol>
