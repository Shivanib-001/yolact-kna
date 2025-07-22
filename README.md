# YOLACT + TensorRT: A Comprehensive Summary
This document provides a detailed overview of how the YOLACT instance segmentation model is used with TensorRT for high-speed inference, along with the postprocessing logic, mask generation strategy, and the considerations for Non-Maximum Suppression (NMS).
## What is YOLACT?
**YOLACT (You Only Look At CoefficienTs)**  is a real-time instance segmentation algorithm that decomposes the task into:
- **Prototype Mask Generation** (global mask features for the whole image)
- **Per-instance Mask Coefficients** (linear combination coefficients)
This allows fast instance segmentation by performing one forward pass and combining masks with coefficients at the end.
## Outputs from YOLACT
A typical YOLACT model outputs the following tensors:
| **Output** | **Description** |
| --- | --- |
| `proto` | Prototype masks: (mask_h, mask_w, mask_dim)
| `loc`   | Bounding box offsets relative to priors
| `mask`  | Linear mask coefficients (num_dets, mask_dim)
| `conf`  | Classification scores (num_dets, num_classes)

