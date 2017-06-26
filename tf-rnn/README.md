## Dependencies

Before you get started, you will need to install tensorflow. Check the official website for how to install it. 

Also, we need tabulate for nice printing

`pip install tabulate`

## Train the network

You can train a new network by running 

`python train.py bounce.csv -folder fancynet`

which writes sampled csv files in `fancynet/generated/` and checkpoints in `fancynet/checkpoints/`. Once your model has written some checkpoints there, you can continue training based on an existing checkpoint (given that the network parameters are the same)

`python train.py bounce.csv -folder fancynet -checkpoint fancynet/checkpoints/E0002L22.36`

where checkpoints have format ExxxxLyyyy where xxxx stands for the epoch and yyyy for the loss

## Sample from the network
During training, the network will write checkpoints at e.g. `fancynet/checkpoints/` which can then be used to sample from the network as follows

`python sample.py randomnet/checkpoints/E0024L-1.681 -amount 10 -layers 2 -hidden 32 -K 7 -seed bounce_seed.csv`

`-checkpoint`, `-layers`, `-hidden`, '-K' (amount of mixtures) are required parameters. The rest is optional. Sample size (N) defaults to 1000, the seed defaults to a zero-seed, and the sampler defaults to greedy. 