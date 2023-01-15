* ref: https://zhuanlan.zhihu.com/p/568769940
* ` nvcc -arch=sm_70 -std=c++14 transpose.cu -o transpose` 
* performance res
    1. naive transpose latency 0.051814 ms, bandwidth 161.90
    2. coalesced transpose latency 0.019350 ms, bandwidth 433.51
    3. coalesced and pad tile leading dim latency 0.011264 ms, bandwidth 744.73
