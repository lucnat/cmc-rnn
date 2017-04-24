## Dependencies

Before you get started, you will need to install tensorflow. Check the official website for how to install it. 

Also, we need tabulate for nice printing

`pip install tabulate`

## Train a new network

You can train a new network by running 

`python mdnlstm.py bounce.csv -folder fancynet`

which writes sampled csv files in `fancynet/generated/` and checkpoints in `fancynet/checkpoints/`. Once your model has written some checkpoints there, you can continue training based on an existing checkpoint (given that the network parameters are the same)

`python mdnlstm.py bounce.csv -folder fancynet -checkpoint fancynet/checkpoints/E0002L22.36`

where checkpoints have format ExxxxLyyyy where xxxx stands for the epoch and yyyy for the loss