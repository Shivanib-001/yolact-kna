import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time


# -------------------- CONFIG --------------------
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)
INPUT_SIZE = 550

COLORS = ((244,  67,  54), (233,  30,  99), (156,  39, 176), (103,  58, 183),
          ( 63,  81, 181), ( 33, 150, 243), (  3, 169, 244), (  0, 188, 212),
          (  0, 150, 136), ( 76, 175,  80), (139, 195,  74), (205, 220,  57),
          (255, 235,  59), (255, 193,   7), (255, 152,   0), (255,  87,  34),
          (121,  85,  72), (158, 158, 158), ( 96, 125, 139))

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# -------------------- PREPROCESS --------------------
def preprocess(img):
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    img = (img - MEANS) / STD
    img = img[:, :, ::-1]  # BGR -> RGB
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0).astype(np.float32)
    return img


# -------------------- PRIORS --------------------
def generate_priors():
    w = h = INPUT_SIZE
    feature_map_sizes = [[int(np.ceil(h/8)), int(np.ceil(w/8))],
                         [int(np.ceil(h/16)), int(np.ceil(w/16))],
                         [int(np.ceil(h/32)), int(np.ceil(w/32))],
                         [int(np.ceil(h/64)), int(np.ceil(w/64))],
                         [int(np.ceil(h/128)), int(np.ceil(w/128))]]

    aspect_ratios = [[1, 0.5, 2]] * len(feature_map_sizes)
    scales = [24, 48, 96, 192, 384]
    priors = []

    for idx, (f_h, f_w) in enumerate(feature_map_sizes):
        scale = scales[idx]
        for y in range(f_h):
            for x in range(f_w):
                cx = (x + 0.5) / f_w
                cy = (y + 0.5) / f_h
                for ratio in aspect_ratios[idx]:
                    r = np.sqrt(ratio)
                    w_p = scale / w * r
                    h_p = scale / h / r
                    priors.append([cx, cy, w_p, h_p])

    return np.array(priors, dtype=np.float32)


def decode(loc, priors, variances=[0.1, 0.2]):
    boxes = np.zeros_like(loc)
    boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    boxes[:, 2:] = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )
    return np.array(indices).flatten() if len(indices) > 0 else np.array([], dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def postprocess(output, original_shape):
    proto, loc, mask, _, conf = output

    loc = np.squeeze(loc, axis=0)
    conf = np.squeeze(conf, axis=0)
    mask = np.squeeze(mask, axis=0)
    proto = np.squeeze(proto, axis=0)

    scores = np.max(conf[:, 1:], axis=1)
    classes = np.argmax(conf[:, 1:], axis=1)
    keep = scores > 0.8

    if not np.any(keep):
        return [], [], [], []

    scores = scores[keep]
    classes = classes[keep]
    mask = mask[keep]
    loc = loc[keep]

    priors = generate_priors()[keep]
    boxes = decode(loc, priors)

    keep_nms = nms(boxes, scores, iou_threshold=0.5)

    boxes = boxes[keep_nms]
    scores = scores[keep_nms]
    classes = classes[keep_nms]
    mask = mask[keep_nms]

    # Generate masks
    masks = proto @ mask.T
    masks = sigmoid(masks)
    masks = np.transpose(masks, (2, 0, 1))

    resized_masks = []
    for m in masks:
        resized = cv2.resize(m, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        resized_masks.append(resized > 0.5)

    masks = np.array(resized_masks, dtype=bool)

    return masks, classes, scores, boxes


# -------------------- ENGINE LOAD & INFER (TRT 8) --------------------
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def infer(engine, image):
    input_data = preprocess(image)
    input_data = np.ascontiguousarray(input_data, dtype=np.float32)
    original_shape = image.shape[:2]

    with engine.create_execution_context() as context:
        # Allocate buffers
        bindings = [None] * engine.num_bindings
        inputs, outputs = [], []
        stream = cuda.Stream()

        for i in range(engine.num_bindings):
            dtype = trt.nptype(engine.get_binding_dtype(i))
            shape = context.get_binding_shape(i)
            size = trt.volume(shape)

            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            bindings[i] = int(device_mem)

            if engine.binding_is_input(i):
                host_mem = np.ascontiguousarray(input_data.ravel(), dtype=dtype)
                cuda.memcpy_htod_async(device_mem, host_mem, stream)
                inputs.append((host_mem, device_mem))
            else:
                host_mem = cuda.pagelocked_empty(size, dtype)
                outputs.append((host_mem, device_mem))

        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Copy outputs
        results = []
        for host_mem, device_mem in outputs:
            cuda.memcpy_dtoh_async(host_mem, device_mem, stream)
            results.append(host_mem)

        stream.synchronize()

        # Reshape outputs
        output_data = []
        out_idx = 0
        for i in range(engine.num_bindings):
            if not engine.binding_is_input(i):
                shape = context.get_binding_shape(i)
                output_data.append(results[out_idx].reshape(shape))
                out_idx += 1

        return output_data, original_shape


# -------------------- MAIN --------------------
if __name__ == "__main__":
    engine_path = "yolact.engine"
    image_path = "../test_images/images.jpeg"

    engine = load_engine(engine_path)
    image = cv2.imread(image_path)

    start_time = time.time()
    outputs, orig_shape = infer(engine, image)
    masks, classes, scores, boxes = postprocess(outputs, orig_shape)
    end_time = time.time()

    print("Inference time:", end_time - start_time, "seconds")

    for i, mask in enumerate(masks):
        color = COLORS[int(classes[i]) % len(COLORS)]
        image[mask] = image[mask] * 0.5 + np.array(color, dtype=np.float32) * 0.5

        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = map(int, [x1 * orig_shape[1], y1 * orig_shape[0],
                                   x2 * orig_shape[1], y2 * orig_shape[0]])
        label = f"{class_names[int(classes[i])]} {scores[i]:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1+5, max(y1 - 5, 0)+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLACT TensorRT Inference", image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

