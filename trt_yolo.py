# trt_yolo.py
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import threading

# IMPORTANT:
# We do NOT use pycuda.autoinit because it binds a CUDA context to the import thread.
# In Flask threaded mode, that can cause crashes / TRT Cask errors.
cuda.init()


class TRTInference:
    def __init__(self, engine_path: str, device_id: int = 0):
        self.logger = trt.Logger(trt.Logger.WARNING)

        self.device = cuda.Device(device_id)
        self.ctx = self.device.make_context()  # dedicated CUDA context
        self.lock = threading.Lock()

        try:
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError("Failed to load TensorRT engine")

            self.context = self.engine.create_execution_context()
            self.inputs = []
            self.outputs = []
            self.stream = cuda.Stream()

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = self.engine.get_tensor_shape(name)
                size = trt.volume(shape)

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append({"name": name, "host": host_mem, "device": device_mem, "shape": shape})
                else:
                    self.outputs.append({"name": name, "host": host_mem, "device": device_mem, "shape": shape})

            # assumes square input
            self.input_size = int(self.inputs[0]["shape"][2])

        finally:
            # pop context after initialization
            self.ctx.pop()

    def infer(self, input_data: np.ndarray):
        # Ensure only one inference runs at a time per TRTInference instance
        with self.lock:
            self.ctx.push()
            try:
                np.copyto(self.inputs[0]["host"], input_data.ravel())
                cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

                for inp in self.inputs:
                    self.context.set_tensor_address(inp["name"], int(inp["device"]))
                for out in self.outputs:
                    self.context.set_tensor_address(out["name"], int(out["device"]))

                self.context.execute_async_v3(stream_handle=self.stream.handle)

                for out in self.outputs:
                    cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

                self.stream.synchronize()
                return [out["host"].reshape(out["shape"]) for out in self.outputs]
            finally:
                self.ctx.pop()

    def close(self):
        # call when exiting app if needed
        try:
            self.ctx.pop()
        except Exception:
            pass
        try:
            self.ctx.detach()
        except Exception:
            pass


class YOLODetectorTRT:
    def __init__(self, engine_path, class_names, conf_threshold=0.5, iou_threshold=0.45):
        self.trt = TRTInference(engine_path)
        self.input_size = self.trt.input_size
        self.class_names = class_names
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.num_classes = len(class_names)

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(self.num_classes, 3), dtype=np.uint8)

        # updated per-frame in preprocess
        self.orig_w = None
        self.orig_h = None
        self.scale = None
        self.pad_x = None
        self.pad_y = None

    def preprocess(self, image):
        self.orig_h, self.orig_w = image.shape[:2]
        self.scale = min(self.input_size / self.orig_w, self.input_size / self.orig_h)
        new_w = int(self.orig_w * self.scale)
        new_h = int(self.orig_h * self.scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)

        self.pad_x = (self.input_size - new_w) // 2
        self.pad_y = (self.input_size - new_h) // 2
        padded[self.pad_y:self.pad_y + new_h, self.pad_x:self.pad_x + new_w] = resized

        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return np.ascontiguousarray(blob)

    def postprocess(self, outputs):
        preds = outputs[0]

        # common TRT export layout: (1, 84, 8400) or (1, 8400, 84)
        if len(preds.shape) == 3:
            if preds.shape[1] < preds.shape[2]:
                preds = preds.transpose(0, 2, 1)
            preds = preds[0]

        boxes, scores, class_ids = [], [], []

        for p in preds:
            x, y, w, h = p[:4]
            class_scores = p[4:4 + self.num_classes]

            cid = int(np.argmax(class_scores))
            conf = float(class_scores[cid])
            if conf < self.conf_threshold:
                continue

            x1 = (x - w / 2 - self.pad_x) / self.scale
            y1 = (y - h / 2 - self.pad_y) / self.scale
            x2 = (x + w / 2 - self.pad_x) / self.scale
            y2 = (y + h / 2 - self.pad_y) / self.scale

            x1 = max(0, min(x1, self.orig_w))
            y1 = max(0, min(y1, self.orig_h))
            x2 = max(0, min(x2, self.orig_w))
            y2 = max(0, min(y2, self.orig_h))

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(cid)

        if not boxes:
            return []

        idx = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        dets = []
        if len(idx) > 0:
            for i in idx.flatten():
                dets.append({
                    "box": [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])],
                    "conf": float(scores[i]),
                    "class_id": int(class_ids[i]),
                })
        return dets

    def detect(self, image_bgr):
        blob = self.preprocess(image_bgr)
        outs = self.trt.infer(blob)
        return self.postprocess(outs)

    def draw(self, image, tracked):
        for t in tracked:
            cid = int(t["class_id"])
            x1, y1, x2, y2 = map(int, t["box"])
            conf = float(t["conf"])
            color = tuple(map(int, self.colors[cid]))
            label = f"ID {t['id']} | {self.class_names[cid]} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(image, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return image

