# VDI on toy dataset
If you have any questions, feel free to pose an issue or send an email to zihao.xu@rutgers.edu. We are always happy to receive feedback!

The code for VDI is developed based on [CIDA](https://github.com/hehaodele/CIDA). [CIDA](https://github.com/hehaodele/CIDA) also provides many baseline implementations (e.g., DANN, MDD), which we used for performance comparasion.

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


## Loss Visualization during training
We use visdom to visualize. We assume the code is run on a remote gpu machine.

### Change Configurations
Find the config in "config" folder. Choose the config you need and Set "opt.use_visdom" to "True".

### Start a Visdom Server on Your Machine
    python -m visdom.server -p 2000
Now connect your computer with the gpu server and forward the port 2000 to your local computer. You can now go to:
    http://localhost:2000 (Your local address)
to see the visualization during training.