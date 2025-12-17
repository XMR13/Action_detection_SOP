import cv2

from yolo_kit import YoloPostConfig, draw_detections, load_class_names, load_pipeline



def read_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    return img


def main():
    path_gambar = "Media/example2.jpg"
    gambar = read_image(path_gambar)

    pipeline = load_pipeline(
        model_path="Models/yolov9-s_v2.onnx",
        post_cfg=YoloPostConfig(conf_threshold=0.45, iou_threshold=0.45)
    )

    class_names = load_class_names("Models/metadata.yaml")

    # deteksi
    detections = pipeline(gambar)
    for det in detections:
        
        print(det.class_id, det.score, det.as_xyxy())

    vis = draw_detections(gambar, detections, class_names=class_names, show_score=True)
    cv2.imshow("detections", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()

    


    
