# VDI on Toy dataset
If you have any questions, feel free to pose an issue or send an email to zihao.xu@rutgers.edu. We are always happy to receive feedback!

The code for VDI is developed based on [CIDA](https://github.com/hehaodele/CIDA). [CIDA](https://github.com/hehaodele/CIDA) also provides many baseline implementations (e.g., DANN, MDD), which we used for performance comparasion in our paper. Please refer to its [code](https://github.com/hehaodele/CIDA) for details.

Notice that we have pretrained the network of q(u|x) and p(x|u) and we included a pretrained weight in this code base. If you want to rerun the pretraining, you need to use the DANN discriminator (set  "opt.d_loss_type = "DANN_loss"" ) and set \sigma_p(x|u) very small (that is, set "opt.lambda_reconstruct" to be about 200~500).

## DG-15
### How to Train on DG-15
    python main.py -c config_DG_15 (or)
    python main.py --config config_DG_15

## DG-60
### How to Train on DG-60
    python main.py -c config_DG_60 (or)
    python main.py --config config_DG_60

## Circle
### How to Train on Circle
    python main.py -c config_Circle (or)
    python main.py --config config_Circle

### Visualization of Circle's Domain Indices
1. Train the VDI on Circle dataset
2. Check your result in "result_save" folder, and then change the first 2 lines in "visualize_circle_indices.py":
```python
dates = "2023-03-10" # filling your own dates for experiments
time = ["14","14","04"] # filling the time for experiments. format: hour, miniute, second
```
3. Run the following code:
```python
python visualize_circle_indices.py
```
Your plot should be in the folder that saves the results of your Circle experiment ("result_save/dates-time/visualization). It will look similar as follows:
<p align="center">
<img src="../fig/visualize_circle.jpg" alt="" data-canonical-src="../fig/visualize_circle.jpg" width="45%"/>
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
