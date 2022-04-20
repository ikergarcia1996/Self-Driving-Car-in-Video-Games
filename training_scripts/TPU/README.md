# TPU Training

Although I have tried to optimize as much as possible the code to run on Google TPUs, 
it seems that TEDD and TPUs do not like each other.

Using a TPU v3-8 VM you will probably find the following error:

```
2022-03-28 11:57:16.078552: E tensorflow/core/tpu/kernels/tpu_compilation_cache_external.cc:113] Computation requires more parameters (3311) than supported (limit 3305).
```

According to [this](https://github.com/pytorch/xla/issues/3453#issuecomment-1083482546) the problem may be resolved in future updates of Pytorch XLA.

Here you can find a discussion about the problem:
https://github.com/pytorch/xla/issues/3453

If you get TEDD running on a TPU please open a pull request or contact me :D

TEDD works fine on CPU and GPU. 