import json
from pathlib import Path

def load_json(path):
    f = open(Path(path))
    data = json.load(f)
    f.close()
    return data

def get_class_labels(path):
    # 'models/class_labels.txt'
    with open(Path(path)) as f:
            labels = [line.strip() for line in f.readlines()]
    return labels

def generate_comment(person_count, class_dict):
    if person_count>0:
        comment = str(person_count) + " Human(s) detected. "
        deliver_person_list = [class_ for class_ in list(class_dict.keys()) if class_ != "other"]
        common_man_list = [class_ for class_ in list(class_dict.keys()) if class_ == "other"]
        print("deliver_person_list : ", deliver_person_list)
        print("common_man_list : ", common_man_list)

        if len(deliver_person_list)>0:
            str_ = ", ".join(deliver_person_list)
            comment = comment + str_ + " delivery person detected."
        if len(common_man_list)>0 and len(deliver_person_list)==0:
             comment = comment + " None is deilvery boy."
             
    else:
        comment = "No Human Detetcted"
    return comment


