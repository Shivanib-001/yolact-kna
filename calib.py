import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, img_dir, cache_file, input_shape, batch_size=1):
        super().__init__()
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.index = 0

        self.device_input = cuda.mem_alloc(
            int(batch_size * np.prod(input_shape) * np.float32().itemsize)
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index + self.batch_size > len(self.img_paths):
            return None

        batch = []
        for i in range(self.batch_size):
            img = cv2.imread(self.img_paths[self.index + i])
            img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            batch.append(img)

        batch = np.ascontiguousarray(batch)
        cuda.memcpy_htod(self.device_input, batch)

        self.index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def generate_calib_cache(onnx_path, calib_img_dir, cache_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = ImageEntropyCalibrator(
        calib_img_dir,
        cache_file,
        input_shape=(3, 550, 550),
        batch_size=1
    )

    print("[INFO] Running calibration only...")
    _ = builder.build_serialized_network(network, config)

    print(f"[SUCCESS] Calibration cache written to {cache_file}")


if __name__ == "__main__":
    generate_calib_cache(
        onnx_path="/home/rnil/Documents/model/yolact-all/yolact-kna/weights/trunk/test.onnx",
        calib_img_dir="images",
        cache_file="yolact_int81.cache"
    )
    
