# VDI on CompCars
If you have any questions, feel free to pose an issue or send an email to zihao.xu@rutgers.edu. We are always happy to receive feedback!

The code for VDI is developed based on [CIDA](https://github.com/hehaodele/CIDA). [CIDA](https://github.com/hehaodele/CIDA) also provides many baseline implementations (e.g., DANN, MDD), which we used for performance comparasion in our paper. Please refer to its [code](https://github.com/hehaodele/CIDA) for details.

In order to eliminate the influence of imbalanced labels, we ensure that each domain shares similar label distributions by picking a subset of [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/). To accelerate training, we derive a 4096-dim feature vector from each input image by Resnet18, and then apply VDI on the feature vectors. These feature vectors are included in "data" folder.

## How to Train on CompCars
    python main.py -c config_CompCars (or)
    python main.py --config config_config_CompCars

## Visualization of CompCars' Domain Indices
1. Train the VDI on CompCars dataset.
2. Check your result in "result_save" folder, and then change the first 2 lines in "visualize_circle_indices.py":
```python
dates = "2023-03-11" # filling your own dates for experiments
time = ["21","59","11"] # filling the time for experiments. format: hour, miniute, second
```
3. Run the following code:
```python
python visualize_compcars_indices.py
```
Your plot should be in the folder that saves the results of your CompCars experiment ("result_save/dates-time/visualization"). It will look similar as follows:
<p align="center">
  <img alt="Light" src="../fig/visualize_compcar_view.jpg" width="50%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="../fig/visualize_compcar_YOM.jpg" width="45%">
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
