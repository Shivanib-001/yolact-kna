import os
import cv2
import time
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pandas as pd
# ================= CONFIG =================

MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)
COLORS = ((244, 67, 54), (233, 30, 99))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

MEANS_T = torch.tensor(MEANS, device="cuda").view(1,3,1,1)
STD_T   = torch.tensor(STD, device="cuda").view(1,3,1,1)

# ================= TORCH GPU PREPROCESS =================

def preprocess_torch(frame, input_size=550):
    # frame: HWC BGR uint8
    x = torch.from_numpy(frame).cuda(non_blocking=True)
    x = x.permute(2,0,1).float().unsqueeze(0)  
    x = F.interpolate(x, (input_size, input_size),
                      mode="bilinear",
                      align_corners=False)
    x = (x - MEANS_T) / STD_T
    x = x[:, [2,1,0]]  # BGR -> RGB
    return x.contiguous()

# ================= LAYER =================

class Layer:
    def __init__(self, score_thresh, iou_thresh, top_k, input_size=550):
        self.SCORE_THRESH = score_thresh
        self.IOU_THRESH = iou_thresh
        self.TOP_K = top_k
        self.INPUT_SIZE = input_size
        self.MASK_THRESH = 0.5
        self.priors = self.generate_priors()
         
   
    def generate_priors(self):
        feature_map_sizes = [[69, 69], [35, 35], [18, 18], [9, 9], [5, 5]]          
        
        w, h = self.INPUT_SIZE,self.INPUT_SIZE

        aspect_ratios = [[1, 0.5, 2]] * len(feature_map_sizes)
        scales = [24, 48, 96, 192, 384]
        priors = []

        for idx, fsize in enumerate(feature_map_sizes):
            scale = scales[idx]
            for y in range(fsize[0]):
                for x in range(fsize[1]):
                    cx = (x + 0.5) / fsize[1]
                    cy = (y + 0.5) / fsize[0]
                    for ratio in aspect_ratios[idx]:
                        r = np.sqrt(ratio)
                        w = scale /self.INPUT_SIZE * r
                        h = scale /self.INPUT_SIZE / r
                        priors.append([cx, cy, w, h])

        return np.array(priors, dtype=np.float32)
    @staticmethod   
    def decode(loc, priors, variances=[0.1,0.2]):

        center_x = priors[:, 0] + loc[:, 0] * variances[0] * priors[:, 2]
        center_y = priors[:, 1] + loc[:, 1] * variances[0] * priors[:, 3]
        width = priors[:, 2] * np.exp(loc[:, 2] * variances[1])
        height = priors[:, 3] * np.exp(loc[:, 3] * variances[1])

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        # Ensure the coordinates are within valid bounds
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        return boxes

    @staticmethod
    def convert_to_xywh(boxes):
        # Convert [x1, y1, x2, y2] -> [x, y, w, h]
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width = x2 - x1
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height = y2 - y1
        boxes_xywh[:, 0] = boxes[:, 0]                # x1
        boxes_xywh[:, 1] = boxes[:, 1]                # y1
        return boxes_xywh

    @staticmethod
    def nms(boxes, scores, iou_threshold, score_thresh, top_k):
        boxes_xywh = Layer.convert_to_xywh(boxes)

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=score_thresh,
            nms_threshold=iou_threshold
        )

        if len(indices) == 0:
            return np.array([], dtype=int)

        indices = np.array(indices).flatten()
        
        # Sort selected indices by score and keep top_k
        sorted_indices = indices[np.argsort(scores[indices])[::-1]]

        return sorted_indices[:top_k]

    @staticmethod
    def sanitize_box(box, img_shape):
        h, w = img_shape
        x1, y1, x2, y2 = box
        return [
            max(0, min(w - 1, x1)),
            max(0, min(h - 1, y1)),
            max(0, min(w - 1, x2)),
            max(0, min(h - 1, y2))
        ]

    # --- Postprocess Function ---
    def postprocess(self,output, orig_shape):
        proto, loc, mask, _, conf = output
        loc = np.squeeze(loc, axis=0)
        conf = np.squeeze(conf, axis=0)
        mask_coeffs = np.squeeze(mask, axis=0)
        proto = np.squeeze(proto, axis=0)

        scores = np.max(conf[:, 1:], axis=1)
        classes = np.argmax(conf[:, 1:], axis=1)
        
        keep = scores > self.SCORE_THRESH
        if not np.any(keep):
            return [], [], [], []

        loc, scores, classes, mask_coeffs = loc[keep], scores[keep], classes[keep], mask_coeffs[keep]
        priors = self.priors[keep]
        
        boxes = Layer.decode(loc, priors)
        keep_nms = Layer.nms(boxes, scores, self.IOU_THRESH, self.SCORE_THRESH, self.TOP_K)
        conf_scores = conf[keep][:, 1:] 
        
        boxes = boxes[keep_nms]
        mask_coeffs = mask_coeffs[keep_nms]
        if len(keep_nms) == 0:
            return [], [], [], []

        masks = 1 / (1 + np.exp(-(proto @ mask_coeffs.T))).transpose(2,0,1)

        

        # Crop in proto space
        ph, pw = proto.shape[:2]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = Layer.sanitize_box([
                int(box[0] * pw), int(box[1] * ph),
                int(box[2] * pw), int(box[3] * ph)
            ], (ph, pw))
            mask = masks[i]
            mask[:y1, :] = mask[y2:, :] = mask[:, :x1] = mask[:, x2:] = 0
            masks[i] = mask

        # Resize and threshold
        result_masks = []
        for mask in masks:
            resized = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
            result_masks.append(resized > self.MASK_THRESH)
        return result_masks, classes.tolist(), scores.tolist(), boxes

# ================= TRT INFERENCE =================

class TRTInference:
    def __init__(self, engine_path):

        cuda.init()
        self.device = cuda.Device(0)
        
        self.cfx = self.device.make_context()  
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = [None] * self.engine.num_io_tensors
        self.device_buffers = {}
        self.host_outputs = {}
        self.input_shape = (1, 3, 550,550)
        self.allocate_buffers()
        
        
        self.cfx.pop()

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def allocate_buffers(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape)
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.device_buffers[name] = device_mem
            self.bindings[i] = int(device_mem)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.host_outputs[name] = np.empty(size, dtype=dtype)

    def infer(self, image):

        self.cfx.push()
        input_tensor = preprocess_torch(image)
        original_shape = image.shape[:2]

        for name in self.device_buffers:
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        for name in self.device_buffers:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, self.input_shape)
                cuda.memcpy_dtod_async(int(self.device_buffers[name]),int(input_tensor.data_ptr()),input_tensor.numel() * input_tensor.element_size(),self.stream)
        
        self.context.execute_async_v3(self.stream.handle)
        
        for name, host_out in self.host_outputs.items():
            cuda.memcpy_dtoh_async(host_out, self.device_buffers[name], self.stream)
            
        self.stream.synchronize()
        
        outputs = []
        for name in sorted(self.host_outputs.keys()):
            shape = self.context.get_tensor_shape(name)
            outputs.append(self.host_outputs[name].reshape(shape))


        self.cfx.pop()
        return outputs, original_shape 


# ================= YOLACT WRAPPER =================

class YOLACT_TRT:
    def __init__(self, engine_path, config_path,
                 score_thresh, iou_thresh, top_k):
        self.model = TRTInference(engine_path)
        self.layer = Layer(score_thresh, iou_thresh, top_k)

    def run(self, frame):
        outputs, shape = self.model.infer(frame)
        return self.layer.postprocess(outputs, shape)

#=================INFERENCE LOGGER===================

class InferenceLogger:
    def __init__(self):
        self.data = []

    def log(self, idx, count, time_ms, fps):
        self.data.append({
            "frame_index": idx,
            "num_detections": count,
            "inference_time_ms": round(time_ms, 2),
            "fps": round(fps, 2)
        })

    def save(self, filename="inference_stats.csv"):
        pd.DataFrame(self.data).to_csv(filename, index=False)
        print(f"Saved inference statistics to {filename}")


# ================= MAIN =================

def main(args):
    yolact = YOLACT_TRT(args.weights, args.config, args.conf_thresh, args.iou_thresh, args.top_k)
    logger = InferenceLogger()
    cap = cv2.VideoCapture(0 if args.video == "webcam" else args.video)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        frame_idx+=1
        if not ret:
            break

        t0 = time.time()
        masks, classes, scores, boxes = yolact.run(frame)
        
        
        for m, b in zip(masks, boxes):
            frame[m] = (0.5 * frame[m] + 0.5 * np.array([0,255,0])).astype(np.uint8)
            x1,y1,x2,y2 = (b * [frame.shape[1],frame.shape[0],
                                frame.shape[1],frame.shape[0]]).astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        fps = 1.0 / (time.time() - t0 + 1e-6)
        inference_time=1000/fps
        cv2.putText(frame, f"FPS: {fps:.2f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        cv2.imshow("YOLACT TRT + TORCH", frame)
        
        logger.log(frame_idx, len(scores), inference_time, fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.save(args.output_csv)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="int8/yolact_trtint8.engine",help="Path to TensorRT engine file")
    parser.add_argument('--video', type=str, default="/home/rnil/Documents/model/yolact-all/test_images/test_video1.mp4", help="Path to video file or 'webcam'")
    parser.add_argument('--config', type=str, default="class_name_kna.txt", help="Path to class names config file (.txt or .json)")
    parser.add_argument('--conf_thresh', type=float, default=0.5, help="Confidence threshold")
    parser.add_argument('--iou_thresh', type=float, default=0.3, help="IoU threshold for NMS")
    parser.add_argument('--top_k', type=int, default=15, help="Maximum number of detections to keep")
    parser.add_argument('--output_csv', type=str, default="inference_stats.csv", help="Path to save inference logs")
    args = parser.parse_args()
    main(args)
