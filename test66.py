import os
import cv2
import time
import numpy as np
import pandas as pd
import argparse
import json

import cv2, numpy as np, time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pandas as pd

# --- Config ---
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)
COLORS = ((244, 67, 54), (233, 30, 99))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class Layer:

    def __init__(self,SCORE_THRESH,IOU_THRESH,TOP_K,INPUT_SIZE=550):
        self.SCORE_THRESH=SCORE_THRESH
        self.IOU_THRESH=IOU_THRESH
        self.top_k=TOP_K
        self.Input_size=INPUT_SIZE
        self.MASK_THRESH = 0.5
        self.priors=self.generate_priors()

    # --- Util Functions ---
    @staticmethod
    def preprocess(img,INPUT_SIZE):
        img = cv2.resize(img, (INPUT_SIZE,INPUT_SIZE)).astype(np.float32)
        img = (img - MEANS) / STD
        img = img[:, :, ::-1].transpose(2, 0, 1)[None, ...]
        return img
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def generate_priors(self):
        feature_map_sizes = [[69, 69], [35, 35], [18, 18], [9, 9], [5, 5]]          
        
        w, h = self.Input_size,self.Input_size

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
                        w = scale / self.Input_size * r
                        h = scale / self.Input_size / r
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
    def _fast_nms(boxes, scores, iou_threshold, score_thresh, top_k):
        """
        Fast NMS implementation based on YOLACT paper
        Parallel processing instead of sequential - much faster!
        """
        # Filter by score threshold first
        valid_mask = scores > score_thresh
        if not np.any(valid_mask):
            return np.array([], dtype=int)
            
        valid_indices = np.where(valid_mask)[0]
        valid_boxes = boxes[valid_indices]
        valid_scores = scores[valid_indices]
        
        # Sort by score (descending) and limit to top_k
        sorted_indices = np.argsort(valid_scores)[::-1][:top_k]
        sorted_boxes = valid_boxes[sorted_indices]
        
        if len(sorted_boxes) == 0:
            return np.array([], dtype=int)
        
        # Calculate IoU matrix for all pairs at once (vectorized)
        # This is the key difference from sequential NMS
        x1 = sorted_boxes[:, 0]
        y1 = sorted_boxes[:, 1] 
        x2 = sorted_boxes[:, 2]
        y2 = sorted_boxes[:, 3]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Broadcast for pairwise IoU calculation
        xx1 = np.maximum(x1[:, None], x1[None, :])
        yy1 = np.maximum(y1[:, None], y1[None, :])
        xx2 = np.minimum(x2[:, None], x2[None, :])
        yy2 = np.minimum(y2[:, None], y2[None, :])
        
        # Calculate intersection areas
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        
        # Calculate IoU matrix
        union = areas[:, None] + areas[None, :] - intersection
        iou_matrix = intersection / (union + 1e-6)
        
        # Fast NMS suppression logic
        # Upper triangular matrix (only compare with higher-scoring boxes)
        triu_mask = np.triu(np.ones_like(iou_matrix, dtype=bool), k=1)
        iou_matrix = iou_matrix * triu_mask
        
        # Find boxes that should be suppressed
        # A box is suppressed if it has high IoU with any higher-scoring box
        suppress_mask = np.any(iou_matrix > iou_threshold, axis=0)
        
        # Keep boxes that are not suppressed
        keep_indices = sorted_indices[~suppress_mask]
        
        # Map back to original indices
        return valid_indices[keep_indices]



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
        #priors.astype(np.float32).tofile('priors.bin')
        boxes = Layer.decode(loc, priors)
        keep_nms = Layer._fast_nms(boxes, scores, self.IOU_THRESH, self.SCORE_THRESH, self.top_k)
        conf_scores = conf[keep][:, 1:]  # shape: [num_dets, num_classes]
        
        boxes = boxes[keep_nms]
        mask_coeffs = mask_coeffs[keep_nms]
        if len(keep_nms) == 0:
            return [], [], [], []
        
        masks = Layer.sigmoid(proto @ mask_coeffs.T).transpose(2, 0, 1)  # (num_dets, proto_h, proto_w)
        

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



# --- TRT Inference Class ---
'''
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
        input_data=image
        input_data = Layer.preprocess(image,550)
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        original_shape = image.shape[:2]

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, self.input_shape)
                cuda.memcpy_htod_async(self.device_buffers[name], input_data, self.stream)
            self.context.set_tensor_address(name, int(self.device_buffers[name]))
        
        self.context.execute_async_v3(self.stream.handle)
        
        for name, host_out in self.host_outputs.items():
            cuda.memcpy_dtoh_async(host_out, self.device_buffers[name], self.stream)
            
        self.stream.synchronize()
        
        outputs = []
        for name in sorted(self.host_outputs.keys()):
            shape = self.context.get_tensor_shape(name)
            outputs.append(self.host_outputs[name].reshape(shape))


        self.cfx.pop()
        return outputs, original_shape '''
class TRTInference:
    def __init__(self, engine_path,INPUT_SIZE=550):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = [None] * self.engine.num_io_tensors
        self.device_buffers = {}
        self.host_outputs = {}
        self.input_size=INPUT_SIZE
        self.input_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
          
        self.allocate_buffers()

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
        input_data = Layer.preprocess(image,self.input_size)
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        original_shape = image.shape[:2]

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, self.input_shape)
                cuda.memcpy_htod_async(self.device_buffers[name], input_data, self.stream)
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        self.context.execute_async_v3(self.stream.handle)

        for name, host_out in self.host_outputs.items():
            cuda.memcpy_dtoh_async(host_out, self.device_buffers[name], self.stream)

        self.stream.synchronize()

        outputs = []
        for name in sorted(self.host_outputs.keys()):
            shape = self.context.get_tensor_shape(name)
            outputs.append(self.host_outputs[name].reshape(shape))
        
        return outputs, original_shape       
        
def class_names_from_config(config_path):
    with open(config_path) as f:    
        return [line.strip() for line in f if line.strip()]


class YOLACT_TRT:
    def __init__(self, engine_path, config_path, score_thresh=0.5, iou_thresh=0.3, top_k=20):
        self.model = TRTInference(engine_path)
        self.class_names = class_names_from_config(config_path)
        self.SCORE_THRESH = score_thresh
        self.IOU_THRESH = iou_thresh
        self.TOP_K = top_k
        self.layer=Layer(self.SCORE_THRESH, self.IOU_THRESH, self.TOP_K)

    def run_inference(self, frame):
        outputs, orig_shape = self.model.infer(frame)

        masks, classes, scores, boxes = self.layer.postprocess(outputs, orig_shape)
        return masks, classes, scores, boxes, orig_shape

    def visualize(self, frame, masks, classes, scores, boxes, orig_shape):
        for mask, cls, score, box in zip(masks, classes, scores, boxes):
            if not np.any(mask):
                continue
            color = COLORS[cls % len(COLORS)]
            overlay = np.zeros_like(frame, dtype=np.uint8)
            overlay[mask] = color
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)

            x1, y1, x2, y2 = map(int, [box[0] * orig_shape[1], box[1] * orig_shape[0], box[2] * orig_shape[1], box[3] * orig_shape[0]])
            label = f"{self.class_names[cls]} {score:.2f}"
            text_w, text_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        return frame

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

def main(args):
    yolact = YOLACT_TRT(args.weights, args.config, args.conf_thresh, args.iou_thresh, args.top_k)
    logger = InferenceLogger()

    cap = cv2.VideoCapture(0 if args.video == "webcam" else args.video)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        masks, classes, scores, boxes, orig_shape = yolact.run_inference(frame)
        inference_time = (time.time() - start_time) * 1000
        fps = 1.0 / ((time.time() - start_time) + 1e-6)

        logger.log(frame_idx, len(scores), inference_time, fps)
        frame = yolact.visualize(frame, masks, classes, scores, boxes, orig_shape)

        cv2.putText(frame, f"Inference: {inference_time:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLACT TensorRT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    logger.save(args.output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', type=str, default="/home/rnil/Documents/model/yolact-all/yolact-kna/weights/trunk/test.engine",help="Path to TensorRT engine file")
    parser.add_argument('--weights', type=str, default="int8/yolact_trtint8.engine",help="Path to TensorRT engine file")
    parser.add_argument('--video', type=str, default="/home/rnil/Documents/model/yolact-all/test_images/test_video1.mp4", help="Path to video file or 'webcam'")
    parser.add_argument('--config', type=str, default="class_name_kna.txt", help="Path to class names config file (.txt or .json)")
    parser.add_argument('--conf_thresh', type=float, default=0.5, help="Confidence threshold")
    parser.add_argument('--iou_thresh', type=float, default=0.3, help="IoU threshold for NMS")
    parser.add_argument('--top_k', type=int, default=15, help="Maximum number of detections to keep")
    parser.add_argument('--output_csv', type=str, default="inference_stats.csv", help="Path to save inference logs")
    args = parser.parse_args()
    main(args)

