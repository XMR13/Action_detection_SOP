import cv2
import numpy as np
import onnxruntime as ort

def letterbox(im, new_shape=(640, 640), color=(114,114,114)):
    h, w = im.shape[:2]
    new_h, new_w = new_shape
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - nw
    pad_h = new_h - nh
    left = pad_w // 2
    top = pad_h // 2

    out = cv2.copyMakeBorder(im_resized, top, pad_h - top, left, pad_w - left,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)

def xyhw_to_xyxy(xyhw):  # YOLOv9 notebook describes [x, y, h, w]
    x, y, h, w = xyhw[...,0], xyhw[...,1], xyhw[...,2], xyhw[...,3]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return np.stack([x1,y1,x2,y2], axis=-1)

def nms_xyxy(boxes, scores, iou_thres=0.45):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2-xx1).clip(0) * (yy2-yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=np.int64)

# --- ONNX inference ---
sess = ort.InferenceSession("Models/best.onnx", providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

img0 = cv2.imread("Media/example3.jpg")              # BGR
img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

img_lb, gain, (padx, pady) = letterbox(img_rgb, (640,640))
x = img_lb.astype(np.float32) / 255.0
x = np.transpose(x, (2,0,1))[None, ...]    # NCHW

out = sess.run(None, {inp_name: x})[0]

# Fix common layout: want (1, N, C)
if out.ndim == 3 and out.shape[1] < out.shape[2]:
    # (1, C, N) -> (1, N, C)
    out = np.transpose(out, (0,2,1))

pred = out[0]  # (N, C)
# C should be 5 + num_classes
boxes_xyhw = pred[:, 0:4]
obj = pred[:, 4:5]
cls = pred[:, 5:]
cls_id = np.argmax(cls, axis=1)
cls_conf = cls[np.arange(cls.shape[0]), cls_id]
conf = (obj[:,0] * cls_conf)

mask = conf > 0.25
boxes_xyxy = xyhw_to_xyxy(boxes_xyhw[mask])
conf = conf[mask]
cls_id = cls_id[mask]

keep = nms_xyxy(boxes_xyxy, conf, iou_thres=0.45)
boxes_xyxy = boxes_xyxy[keep]
conf = conf[keep]
cls_id = cls_id[keep]

# Undo letterbox to original image coords
boxes_xyxy[:, [0,2]] -= padx
boxes_xyxy[:, [1,3]] -= pady
boxes_xyxy /= gain

# Draw on original (BGR)
for (x1,y1,x2,y2), s, c in zip(boxes_xyxy, conf, cls_id):
    x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
    cv2.rectangle(img0, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img0, f"{c}:{s:.2f}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imwrite("vis.jpg", img0)
