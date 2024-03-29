# VDI on TPT-48
If you have any questions, feel free to pose an issue or send an email to zihao.xu@rutgers.edu. We are always happy to receive feedback!

The code for VDI is developed based on [CIDA](https://github.com/hehaodele/CIDA). [CIDA](https://github.com/hehaodele/CIDA) also provides many baseline implementations (e.g., DANN, MDD), which we used for performance comparasion in our paper. Please refer to its [code](https://github.com/hehaodele/CIDA) for details.

We approximate the Earth Moving distance with [Sinkhorn Loss](https://audeg.github.io/talks/talkLiege.pdf) to accelerate computation. This is done by using the [GeomLoss](https://www.kernel-operations.io/geomloss/) package.

## W $\rightarrow$ E
### How to Train for task W $\rightarrow$ E on TPT-48
    python main.py -c config_TPT_48_WE (or)
    python main.py --config config_TPT_48_WE
### How to Use the Pretrained Model to Do Inference
1. Download the weight from [here](https://drive.google.com/drive/folders/1TNkak042AsGHJQ9hoQKOzubU43YQlXDg?usp=sharing) and unzip under the folder "pretrain_weight".
2. Run the following code:
```python
    python inference.py -c config_TPT_48_WE_inference (or)
    python inference.py --config config_TPT_48_WE_inference
```
Both training/inference will produce the result in "result_save" folder, and you can use the result to do visualization.

## N $\rightarrow$ S
### How to Train for task N $\rightarrow$ S on TPT-48
    python main.py -c config_TPT_48_NS (or)
    python main.py --config config_TPT_48_NS
### How to Use the Pretrained Model to Do Inference
1. Download the weight from [here](https://drive.google.com/drive/folders/1TNkak042AsGHJQ9hoQKOzubU43YQlXDg?usp=sharing) and unzip under the folder "pretrain_weight".
2. Run the following code:
```python
    python inference.py -c config_TPT_48_NS_inference (or)
    python inference.py --config config_TPT_48_NS_inference
```
Both training/inference will produce the result in "result_save" folder, and you can use the result to do visualization.


### Visualization of TPT-48's Domain Indices
1. Train/inference the VDI on TPT-48 with either "W $\rightarrow$ E" or "N $\rightarrow$ S" task.
2. Check your result in "result_save" folder, and then change the first 2 lines in "visualize_tpt_48_indices.py":
```python
dates = "2023-03-11" # filling your own dates for experiments
time = ["12","56","54"] # filling the time for experiments. format: hour, miniute, second
```
3. Run the following code:
```python
python visualize_tpt_48_indices.py
```
Your plot should be in the folder that saves the results of your TPT-48 experiment ("result_save/dates-time/visualization"). It will look similar as follows:
<p align="center">
<img src="../fig/visualize_tpt_48.jpg" alt="" data-canonical-src="../fig/visualize_tpt_48.jpg" width="93%"/>
</p>


## Loss Visualization during Training
We use visdom to visualize. We assume the code is run on a remote gpu machine.

### Change Configurations
Find the config in "config" folder. Choose the config you need and Set "opt.use_visdom" to "True".

### Start a Visdom Server on Your Machine
    python -m visdom.server -p 2000
Now connect your computer with the gpu server and forward the port 2000 to your local computer. You can now go to:
    http://localhost:2000 (Your local address)
to see the visualization during training.
