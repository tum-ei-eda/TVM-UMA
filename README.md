# TVM-UMA
This repository was added to test the UMA functionality. 

There are the code structure for Vanilla accelerator backend as well as the python codes for running three examples:
1. Conv2D 
2. MobileNet
3. TFLite

This is the reference link that is used to create the code skelton of Vanilla accelerator. 

(Vanilla)[https://tvm.apache.org/docs/tutorial/uma.html]

For running the TFLite models on Vanilla accelerator, you need to add a **relay pass** to the backend to convert memory layout. 
