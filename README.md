# TVM-UMA
This repository was added to test the UMA functionality. 

There are the code structure for integration of Vanilla accelerator into TVM backends as well as the python codes for running three examples:
1. Conv2D (run_conv2d.py)
2. MobileNet (run_mobilenet.py)
3. TFLite (run_tflite.py)

This is the reference link that is used to create the code skeltons of Vanilla accelerator. 
[https://tvm.apache.org/docs/tutorial/uma.html]

The quantized_vanilla_example branch contains the quantized version of Vanilla (QVanilla).

**Note:** For running the TFLite models on Vanilla accelerator, you need to add a **relay pass** to the backend to convert memory layout. 
[https://tvm.apache.org/docs/arch/convert_layout.html]
