import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

MEANS = np.array([103.94, 116.78, 123.68], dtype=np.float32)
STD   = np.array([57.38, 57.12, 58.40], dtype=np.float32)

class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_dir,
        cache_file,
        batch_size=1,
        input_shape=(3, 550, 550),
        max_images=100
    ):
        super().__init__()

        self.image_dir = image_dir
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape

        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ][:max_images]

        self.current_index = 0
       
        self.device_input = cuda.mem_alloc(
           int(batch_size * np.prod(input_shape) * np.float32().itemsize)
        )

    def get_batch_size(self):
        return self.batch_size

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
        img = img.astype(np.float32)

        img = (img - MEANS) / STD
        img = img[:, :, ::-1]          # BGR → RGB
        img = img.transpose(2, 0, 1)   # CHW
        return img

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_paths):
            return None

        batch = np.zeros(
            (self.batch_size, *self.input_shape),
            dtype=np.float32
        )

        for i in range(self.batch_size):
            batch[i] = self.preprocess(
                self.image_paths[self.current_index + i]
            )

        self.current_index += self.batch_size

        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[INFO] Using calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[INFO] Calibration cache saved: {self.cache_file}")

def build_int8_engine(
    onnx_path,
    engine_path,
    calib_image_dir,
    calib_cache,
    input_name="input",
    input_shape=(1, 3, 550, 550),
    batch_size=1,
    workspace_size_gb=4
):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(
             1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
         ) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size_gb << 30
        )

        # Enable INT8 + FP16
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)

        calibrator = ImageEntropyCalibrator(
            image_dir=calib_image_dir,
            cache_file=calib_cache,
            batch_size=batch_size,
            input_shape=input_shape[1:]
        )
        config.int8_calibrator = calibrator

        print("[INFO] Parsing ONNX...")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parsing failed")

        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_name,
            input_shape,
            input_shape,
            input_shape
        )
        config.add_optimization_profile(profile)

        print("[INFO] Building INT8 engine...")
        

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("INT8 engine build failed")

        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)

        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

        print(f"[SUCCESS] INT8 engine saved to {engine_path}")


        



if __name__ == "__main__":
    ONNX_PATH = "/home/rnil/Documents/model/yolact-all/yolact-kna/weights/trunk/test.onnx"
    ENGINE_PATH = "yolact_int8.engine"
    CALIB_IMAGES = "images"
    CALIB_CACHE = "yolact_int8.cache"

    build_int8_engine(
        onnx_path=ONNX_PATH,
        engine_path=ENGINE_PATH,
        calib_image_dir=CALIB_IMAGES,
        calib_cache=CALIB_CACHE,
        batch_size=1
    )

