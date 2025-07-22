# YOLACT + TensorRT: A Comprehensive Summary
This document provides a detailed overview of how the YOLACT instance segmentation model is used with TensorRT for high-speed inference, along with the postprocessing logic, mask generation strategy, and the considerations for Non-Maximum Suppression (NMS).
## What is YOLACT?
**YOLACT (You Only Look At CoefficienTs)**  is a real-time instance segmentation algorithm that decomposes the task into:
- **Prototype Mask Generation** (global mask features for the whole image)
- **Per-instance Mask Coefficients** (linear combination coefficients)
This allows fast instance segmentation by performing one forward pass and combining masks with coefficients at the end.
## Model Conversion ##
To convert a machine learning model to ONNX (Open Neural Network Exchange) format, you typically need to use a framework-specific converter. For example, you can use tf2onnx to convert TensorFlow models, torch.onnx.export for PyTorch models, and skl2onnx for scikit-learn models. The general process involves loading the model, defining input shapes, and then using the converter to generate the ONNX file. 
### Converting the model.pth to model.onnx ###
In this section we have following python script which are having dependencies on Yolact main script i.e.
> - **yolact_onnx.py**
> - **path_to_onnx.py**

To convert the .pth file to the .onnx format just run the following command:
> `python3 pth_to_onnx.py --trained_model=weights/yolact_resnet50_54_800000.pth`

## Outputs from YOLACT
A typical YOLACT model outputs the following tensors:
| **Output** | **Description** |
| --- | --- |
| `proto` | Prototype masks: (mask_h, mask_w, mask_dim)
| `loc`   | Bounding box offsets relative to priors
| `mask`  | Linear mask coefficients (num_dets, mask_dim)
| `conf`  | Classification scores (num_dets, num_classes)

These are decoded into bounding boxes, classes, scores, and instance masks.
## Decoding Bounding Boxes ##
YOLACT uses prior boxes (anchors) and predicts offsets to them:
```
center_x = prior_cx + loc_dx * variance[0] * prior_w
center_y = prior_cy + loc_dy * variance[0] * prior_h
width    = prior_w * exp(loc_dw * variance[1])
height   = prior_h * exp(loc_dh * variance[1])
```
The final boxes are converted from center format to `[x1, y1, x2, y2]`.
## Prior Box Generation ##
YOLACT uses priors at multiple feature map resolutions. Typical settings:
- **Feature maps:** [69x69, 35x35, 18x18, 9x9, 5x5]
- **Scales:** [24, 48, 96, 192, 384]
- **Aspect Ratios:** [1.0, 0.5, 2.0]
Each feature location generates multiple anchor boxes of different aspect ratios.
## Postprocessing Steps ##
- **Confidence Filtering:**
  - Filter out detections with score below threshold
- **Decode Bounding Boxes:**
  - Use priors + predicted offsets to compute real box coordinates
- **Non-Maximum Suppression (NMS):**
  - Remove overlapping detections
  - Uses OpenCV - `cv2.dnn.NMSBoxes`
- **Generate Masks:**
  - Combine `proto` and `mask_coeffs`:
  - 'masks = sigmoid(proto @ mask_coeffs.T).transpose(2, 0, 1)`
    - Crop, resize and threshold to get binary masks
## TensorRT Acceleration ##
**TensorRT** is a high-performance deep learning inference engine developed by NVIDIA. It supports optimizing trained models to run efficiently on NVIDIA GPUs.

**_Key Steps:_**
- **Export YOLACT model to ONNX**
- **Build a TensorRT engine from ONNX**
- **Run inference using GPU & Cuda**

