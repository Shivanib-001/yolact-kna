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
    x = x.permute(2,0,1).float().unsqueeze(0)  # 1x3xHxW
    x = F.interpolate(x, (input_size, input_size),
                      mode="bilinear",
                      align_corners=False)
    x = (x - MEANS_T) / STD_T
    x = x[:, [2,1,0]]  # BGR -> RGB
    return x.contiguous()

# ================= LAYER (CPU POSTPROCESS, UNCHANGED) =================

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
                        w = scale / self.INPUT_SIZE * r
                        h = scale / self.INPUT_SIZE/ r
                        priors.append([cx, cy, w, h])

        return np.array(priors, dtype=np.float32)



    @staticmethod
    def decode(loc, priors, variances=[0.1,0.2]):
        cx = priors[:,0] + loc[:,0] * variances[0] * priors[:,2]
        cy = priors[:,1] + loc[:,1] * variances[0] * priors[:,3]
        w  = priors[:,2] * np.exp(loc[:,2] * variances[1])
        h  = priors[:,3] * np.exp(loc[:,3] * variances[1])
        return np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)

    @staticmethod
    def nms(boxes, scores, thresh, score_thresh, top_k):
        idx = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_thresh,
            thresh
        )
        if len(idx) == 0:
            return []
        idx = np.array(idx).flatten()
        return idx[:top_k]

    def postprocess(self, outputs, orig_shape):
        proto, loc, mask, _, conf = outputs
        proto = proto.squeeze(0)
        loc = loc.squeeze(0)
        conf = conf.squeeze(0)
        mask = mask.squeeze(0)

        scores = np.max(conf[:,1:], axis=1)
        classes = np.argmax(conf[:,1:], axis=1)
        keep = scores > self.SCORE_THRESH
        if not np.any(keep):
            return [], [], [], []

        loc, scores, classes, mask = loc[keep], scores[keep], classes[keep], mask[keep]
        priors = self.priors[keep]

        boxes = self.decode(loc, priors)
        keep_idx = self.nms(boxes, scores, self.IOU_THRESH, self.SCORE_THRESH, self.TOP_K)

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        classes = classes[keep_idx]
        mask = mask[keep_idx]

        masks = 1 / (1 + np.exp(-(proto @ mask.T))).transpose(2,0,1)

        h, w = orig_shape
        final_masks = []
        for m in masks:
            m = cv2.resize(m, (w,h))
            final_masks.append(m > self.MASK_THRESH)

        return final_masks, classes.tolist(), scores.tolist(), boxes

# ================= TRT INFERENCE =================

class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = {}
        self.device_buffers = {}
        self.input_shape = (1,3,550,550)

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            buf = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.device_buffers[name] = buf
            self.bindings[name] = i

        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
               == trt.TensorIOMode.OUTPUT
        ]

    def load_engine(self, path):
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            return rt.deserialize_cuda_engine(f.read())

    def infer(self, frame):
        input_tensor = preprocess_torch(frame)
        orig_shape = frame.shape[:2]

        try:
            for name in self.device_buffers:
                self.context.set_tensor_address(name, int(self.device_buffers[name]))

            for name in self.device_buffers:
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.context.set_input_shape(name, self.input_shape)
                    cuda.memcpy_dtod_async(
                        int(self.device_buffers[name]),
                        int(input_tensor.data_ptr()),
                        input_tensor.numel() * input_tensor.element_size(),
                        self.stream
                    )

            self.context.execute_async_v3(self.stream.handle)

            outputs = []
            for name in sorted(self.output_names):
                shape = self.context.get_tensor_shape(name)
                size = trt.volume(shape)
                host = np.empty(size, dtype=np.float32)
                cuda.memcpy_dtoh_async(host, self.device_buffers[name], self.stream)
                outputs.append(host.reshape(shape))

            self.stream.synchronize()
            return outputs, orig_shape

        finally:
            pass

# ================= YOLACT WRAPPER =================

class YOLACT_TRT:
    def __init__(self, engine_path, config_path,
                 score_thresh, iou_thresh, top_k):
        self.model = TRTInference(engine_path)
        self.layer = Layer(score_thresh, iou_thresh, top_k)

    def run(self, frame):
        outputs, shape = self.model.infer(frame)
        return self.layer.postprocess(outputs, shape)

# ================= MAIN =================

def main(args):
    yolact = YOLACT_TRT(args.weights, args.config, args.conf_thresh, args.iou_thresh, args.top_k)

    cap = cv2.VideoCapture(0 if args.video == "webcam" else args.video)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        masks, classes, scores, boxes = yolact.run(frame)
        fps = 1.0 / (time.time() - t0 + 1e-6)

        for m, b in zip(masks, boxes):
            frame[m] = (0.5 * frame[m] + 0.5 * np.array([0,255,0])).astype(np.uint8)
            x1,y1,x2,y2 = (b * [frame.shape[1],frame.shape[0],
                                frame.shape[1],frame.shape[0]]).astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame, f"FPS: {fps:.2f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        cv2.imshow("YOLACT TRT + TORCH", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="/home/rnil/Documents/model/yolact-all/yolact-kna/weights/trunk/test.engine",help="Path to TensorRT engine file")
    parser.add_argument('--video', type=str, default="/home/rnil/Documents/model/yolact-all/test_images/test_video1.mp4", help="Path to video file or 'webcam'")
    parser.add_argument('--config', type=str, default="class_name_kna.txt", help="Path to class names config file (.txt or .json)")
    parser.add_argument('--conf_thresh', type=float, default=0.5, help="Confidence threshold")
    parser.add_argument('--iou_thresh', type=float, default=0.3, help="IoU threshold for NMS")
    parser.add_argument('--top_k', type=int, default=15, help="Maximum number of detections to keep")
    parser.add_argument('--output_csv', type=str, default="inference_stats.csv", help="Path to save inference logs")
    args = parser.parse_args()
    main(args)

