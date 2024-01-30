from ultralytics import YOLO
import pandas as pd
from PIL import Image
from pathlib import Path

def load_object_detector():
    model_path = Path("models/object_detector_best.pt")
    model = YOLO(model_path)
    return model

def get_object_image(result_df, img_path):
    object_path_list = []
    for idx, row in result_df.iterrows():
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        output_image_path = "static/outputs/img_" + str(idx) + ".jpg"
        object_path_list.append(output_image_path)
        try:
            with Image.open(img_path) as img:
                cropped_img = img.crop((xmin, ymin, xmax, ymax))
                cropped_img.save(output_image_path, 'JPEG')
                print(f"Cropped image saved to: {output_image_path}")
        except Exception as e:
            print(f"Error cropping and saving image: {e}")
    return object_path_list

def get_object_detection(model, img_path):
    output_json = {}
    result = model.predict(img_path, iou=0.3)

    result_df = pd.DataFrame(columns=["name", "xmin", "ymin", "xmax", "ymax", "confidence"])
    for box in result[0].boxes:
        class_id = result[0].names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        xmin, ymin, xmax, ymax = cords

        result_df.loc[len(result_df)] = [class_id, xmin, ymin, xmax, ymax, conf]
    result_df['confidence'] = pd.to_numeric(result_df['confidence'])
    object_path_list = get_object_image(result_df, img_path)
    # output_json["result"] = result_df
    output_json["person_count"] = len(result_df)
    output_json["object_list"] = object_path_list
    return output_json




# img_path = r"D:\Study\JOBS\HAVELLS\havels_local_project\static\uploads\Uber eats Delivery Boy_6.jpg"
# result_df = get_object_classification(img_path)
# print(result_df)