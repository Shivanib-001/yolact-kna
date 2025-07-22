import cv2, numpy as np, time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pandas as pd


''' --------------------------- CONFIGURATION ---------------------------'''

# Mean and std for image normalization (used during training on ImageNet)
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)

# Input size expected by the YOLACT model
INPUT_SIZE = 550

# Thresholds and limits for filtering and display
SCORE_THRESH = 0.1        
IOU_THRESH = 0.3         
MASK_THRESH = 0.5      
top_k = 10          

# Color map for mask visualization taken from yolact config
COLORS = ((244, 67, 54), (233, 30, 99), (156, 39, 176), (103, 58, 183),
          (63, 81, 181), (33, 150, 243), (3, 169, 244), (0, 188, 212),
          (0, 150, 136), (76, 175, 80), (139, 195, 74), (205, 220, 57),
          (255, 235, 59), (255, 193, 7), (255, 152, 0), (255, 87, 34),
          (121, 85, 72), (158, 158, 158), (96, 125, 139))

# List of class names
class_names = ['trunk']

# TensorRT logger for errors/warnings
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


'''--------------------------- PREPROCESSING ---------------------------'''

def preprocess(img):
    '''Preprocess the image to match model input: resize, normalize, format'''
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    img = (img - MEANS) / STD                            # Normalize
    img = img[:, :, ::-1].transpose(2, 0, 1)[None, ...]  # BGR → RGB and NCHW
    return img

def sigmoid(x):
    '''
        Converts raw model outputs ("logits") into values in the range [0, 1].
        This is essential for interpreting outputs as probabilities
    '''
    return 1 / (1 + np.exp(-x))


'''--------------------------- PRIORS ---------------------------'''


def generate_priors():
    '''
    # Generate anchor boxes (priors) used for decoding predicted boxes
    Feature map sizes : 
        # YOLACT uses priors at multiple scales and aspect ratios.
        # Each feature map level detects objects at a specific scale and applies a set of anchors.
        # The following feature map sizes correspond to FPN layers:
        # C3 (69x69), C4 (35x35), C5 (18x18), P6 (9x9), P7 (5x5)
        # These sizes are based on the 550x550 input and downsampling stride of 8, 16, 32,...
    
    Aspect ratio :
        # For each anchor, YOLACT uses 3 aspect ratios: 1:1 (square), 1:2 (tall), 2:1 (wide)
        # These cover a wide range of object shapes.  
    
    Scales:
        # Anchor box base sizes for each feature map level.
        # These are the scales of the boxes relative to the 550 input 
        scales = [24, 48, 96, 192, 384]

    Note that priors are [x,y,width,height] where (x,y) is the center of the box  
    '''

    feature_map_sizes = [[69, 69], [35, 35], [18, 18], [9, 9], [5, 5]]

    aspect_ratios = [[1, 0.5, 2]] * len(feature_map_sizes)
       
    scales = [24, 48, 96, 192, 384]  # in pixels

    # Initialize list of priors (cx, cy, w, h) format, all relative to [0,1]
    priors = []

    # Loop over each feature map level
    for idx, fsize in enumerate(feature_map_sizes):
        scale = scales[idx] # scales corresponding to this level (depending on FPN layer)

        # For each cell in the feature map grid
        for y in range(fsize[0]):
            for x in range(fsize[1]):
                # Compute the center (cx, cy) of the anchor in normalized coordinates
                # Adding 0.5 places the anchor at the center of the cell, not the top-left corner
                cx = (x + 0.5) / fsize[1]
                cy = (y + 0.5) / fsize[0]

                # For each aspect ratio, compute corresponding anchor dimensions
                for ratio in aspect_ratios[idx]:
                    r = np.sqrt(ratio)
                    # Normalize width and height by input size (550)
                    w = scale / INPUT_SIZE * r
                    h = scale / INPUT_SIZE / r
                    priors.append([cx, cy, w, h])

    return np.array(priors, dtype=np.float32)

''' --------------------------- BOX DECODE ---------------------------'''

def decode(loc, priors, variances=[0.1, 0.2]):
    '''Decodes the predicted localization offsets back into actual bounding box coordinates. '''

    # Recover the center x and y of the predicted box
    center_x = priors[:, 0] + loc[:, 0] * variances[0] * priors[:, 2]
    center_y = priors[:, 1] + loc[:, 1] * variances[0] * priors[:, 3]

    # Recover the width and height using exponential decoding
    # This ensures predicted widths/heights are always positive
    width = priors[:, 2] * np.exp(loc[:, 2] * variances[1])
    height = priors[:, 3] * np.exp(loc[:, 3] * variances[1])

    # Convert (cx, cy, w, h) → (x1, y1, x2, y2)
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    return boxes


def convert_to_xywh(boxes):
    '''Converts bounding boxes from corner format [x1, y1, x2, y2] to top-left format [x, y, w, h]. ''' 
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width = x2 - x1
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height = y2 - y1
    boxes_xywh[:, 0] = boxes[:, 0]                # x1
    boxes_xywh[:, 1] = boxes[:, 1]                # y1
    return boxes_xywh

 
def nms(boxes, scores, iou_threshold, score_thresh, top_k):
    '''Apply Non-Maximum Suppression (NMS) using OpenCV'''
    boxes_xywh = convert_to_xywh(boxes)

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh.tolist(),
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


def sanitize_box(box, img_shape):
    '''Clamp box coordinates to image dimensions. Ensures that bounding box coordinates are within the valid image range.'''
    
    h, w = img_shape
    x1, y1, x2, y2 = box
    return [
        max(0, min(w - 1, x1)),
        max(0, min(h - 1, y1)),
        max(0, min(w - 1, x2)),
        max(0, min(h - 1, y2))
    ]

''' --------------------------- POSTPROCESS FUNCTION ---------------------------'''

def postprocess(output, orig_shape):
    '''
    Post-processes the raw outputs from the YOLACT model.

    Args:
        output (list of np.ndarrays): List of outputs from the network in the order:
            proto (mask bases), loc (box regressions), mask (mask coefficients),
            (unused), conf (class confidences)
        orig_shape (tuple): Original image shape (height, width), used for resizing masks.

    Returns:
        Tuple: (result_masks, classes, scores, boxes)
            - result_masks: list of boolean masks resized to original image size
            - classes: list of class indices for each detection
            - scores: list of confidence scores
            - boxes: list of bounding boxes in normalized [x1, y1, x2, y2] format
    '''

    # Unpack output tensors from the model
    proto, loc, mask, _, conf = output

    # Remove batch dimension (since inference is for a single image)
    loc = np.squeeze(loc, axis=0)             
    conf = np.squeeze(conf, axis=0)           
    mask_coeffs = np.squeeze(mask, axis=0)    
    proto = np.squeeze(proto, axis=0)        

    # Step 1: Classification
    # Take the best scoring class (excluding background, index 0)
    scores = np.max(conf[:, 1:], axis=1)     
    classes = np.argmax(conf[:, 1:], axis=1) 

    # Step 2: Confidence Thresholding
    # Filter out detections below confidence threshold
    keep = scores > SCORE_THRESH
    if not np.any(keep):
        return [], [], [], [] 

    # Apply mask to keep only high-confidence predictions
    loc = loc[keep]
    scores = scores[keep]
    classes = classes[keep]
    mask_coeffs = mask_coeffs[keep]

    # Step 3: Box Decoding
    # Generate matching priors for filtered predictions
    priors = generate_priors()[keep]
    # Decode loc into (x1, y1, x2, y2) format
    boxes = decode(loc, priors)

    # Step 4: Non-Maximum Suppression
    # Remove overlapping detections for the same class
    keep_nms = nms(boxes, scores, IOU_THRESH, SCORE_THRESH, top_k)
    if len(keep_nms) == 0:
        return [], [], [], []

    # Keep only NMS-filtered detections
    boxes = boxes[keep_nms]
    scores = scores[keep_nms]
    classes = classes[keep_nms]
    mask_coeffs = mask_coeffs[keep_nms]

    # Step 5: Mask Generation
    # The key innovation of YOLACT: generate masks on-the-fly using linear combination
    # of prototype masks and per-instance coefficients
    # Masks = sigmoid(PC)T 
    # 
    # p = proto shape: [H, W, num_protos]
    # c = mask_coeffs.T shape: [num_protos, num_dets]
    # result: [H, W, num_dets] → transpose to [num_dets, H, W]
    masks = sigmoid(proto @ mask_coeffs.T).transpose(2, 0, 1)

    # Step 6: Crop masks in prototype space using predicted bounding boxes
    ph, pw = proto.shape[:2]  # Prototype mask dimensions
    for i, box in enumerate(boxes):
        # Convert normalized box coordinates to proto mask dimensions
        x1, y1, x2, y2 = sanitize_box([
            int(box[0] * pw), int(box[1] * ph),
            int(box[2] * pw), int(box[3] * ph)
        ], (ph, pw))

        mask = masks[i]
        mask[:y1, :] = mask[y2:, :] = mask[:, :x1] = mask[:, x2:] = 0
        masks[i] = mask

    # Step 7: Resize masks to match original image size and apply threshold
    result_masks = []
    for mask in masks:
        # Resize from prototype space to original image resolution
        resized = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
        # Apply threshold to convert soft mask → binary mask
        result_masks.append(resized > MASK_THRESH)

    # Final output: masks, class IDs, scores, and bounding boxes
    return result_masks, classes.tolist(), scores.tolist(), boxes

''' --------------------------- TRT INFERENCE CLASS ---------------------------'''

class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = [None] * self.engine.num_io_tensors
        self.device_buffers = {}
        self.host_outputs = {}
        self.input_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)

        self.allocate_buffers()
    
    # Load serialized engine
    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    # Allocate GPU memory for all input/output tensors
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
                
    # Run inference on preprocessed image
    def infer(self, image):
        input_data = preprocess(image)
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        original_shape = image.shape[:2]
        
        # Copy data to input buffer and set context
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, self.input_shape)
                cuda.memcpy_htod_async(self.device_buffers[name], input_data, self.stream)
            self.context.set_tensor_address(name, int(self.device_buffers[name]))
        
        # Launch inference
        self.context.execute_async_v3(self.stream.handle)
        
        # Retrieve outputs from GPU
        for name, host_out in self.host_outputs.items():
            cuda.memcpy_dtoh_async(host_out, self.device_buffers[name], self.stream)

        self.stream.synchronize()
        
        # Convert flat output buffers to correct shapes
        outputs = []
        for name in sorted(self.host_outputs.keys()):
            shape = self.context.get_tensor_shape(name)
            outputs.append(self.host_outputs[name].reshape(shape))

        return outputs, original_shape


# --- Main Loop ---
if __name__ == "__main__":
    trt_infer = TRTInference("model/kna_trunk.engine")
    cap = cv2.VideoCapture("../test_images/test_video3.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open camera")
        exit()
        
    log_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()

        # Run inference and postprocessing
        outputs, orig_shape = trt_infer.infer(frame)
        masks, classes, scores, boxes = postprocess(outputs, orig_shape)

        # Calculate FPS and inference time
        inference_time = (time.time() - start_time) * 1000
        fps = 1.0 / ((time.time() - start_time) + 1e-6)
        
        num_detections = len(scores)
        # Log statistics
        log_data.append({
            "frame_index": frame_idx,
            "num_detections": num_detections,
            "inference_time_ms": round(inference_time, 2),
            "fps": round(fps, 2)
        })
        frame_idx += 1

        # Draw detections
        for mask, cls, score, box in zip(masks, classes, scores, boxes):
            if not np.any(mask):
                continue
            color = COLORS[cls % len(COLORS)]
            overlay = np.zeros_like(frame, dtype=np.uint8)
            overlay[mask] = color
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)
            x1, y1, x2, y2 = map(int, [box[0] * orig_shape[1], box[1] * orig_shape[0],box[2] * orig_shape[1], box[3] * orig_shape[0]])
            label = f"{class_names[cls]} {score:.2f}"
            text_w, text_h = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX , 0.6,1)[0]
            text_pt = (x1 + 5, y1 - 10)
            cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 10, y1),color, -1)
            cv2.putText(frame, label, text_pt,cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            
        cv2.putText(frame, f"Inference: {inference_time:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f"YOLACT TensorRT - Webcam, SCORE_THRESH = {SCORE_THRESH} IOU_THRESH={IOU_THRESH} ", frame.astype(np.uint8))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save stats to CSV
    df = pd.DataFrame(log_data)
    df.to_csv("inference_stats_KNA.csv", index=False)
    print("Saved inference statistics to inference_stats.csv")
    
