# GPU Training

These are the scripts used to train the released models. 
You can use them to replicate the results or to train your own models.

The scripts are `slurm batch files`, they were run in a system with A100 GPUs (80Gb) and 1024Gb of RAM.
But you should be able to train the `TEDD_1140_s` and `TEDD_1140_m` models in most modern GPUs with at least 8GB of RAM and FP16 support. 
You will probably need at least 32GB of RAM, if your system runs out of memory or freezes try reducing
the `--dataloader_num_workers` parameter. 

You can use the `evaluate_models.sh` script to evaluate the models on the test set.

If you have any problem, you find any bug, our you find a set of parameters that works better than the ones I used, please open an issue or contact me :smiley:

