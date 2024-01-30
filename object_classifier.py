import torch
import torch.nn as nn
from torchvision import  transforms, models
from PIL import Image

def get_class_labels():
    with open('models\class_labels.txt') as f:
            labels = [line.strip() for line in f.readlines()]
    return labels

def load_object_classifier(labels):
    
    # Create a custom MobileNetV2 model class
    class CustomMobileNetV2(nn.Module):
        def __init__(self, num_classes):
            super(CustomMobileNetV2, self).__init__()
            self.base_model = models.mobilenet_v2(pretrained=False)
            self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

        def forward(self, x):
            return self.base_model(x)
    # Initialize the custom model
    model = CustomMobileNetV2(num_classes=len(labels))
    # Load the best model weights
    model.load_state_dict(torch.load('models\image_classifier.pth', map_location=torch.device('cpu')))  
    return model

def get_image_class(model, labels, image_path):
    # Load the saved class labels
    # Set the model to evaluation mode
    model.eval()

    # Image preprocessing for inference
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the input image (adjust the image path)
    # image_path = 'tgoogle\images\cropped_images\Dunzo Delivery Boy_107_dunzo_4.jpg'
    image = Image.open(image_path)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()

    # Get the predicted class label
    predicted_class_label = labels[predicted_class_index]
    print("predicted_class_label : ", predicted_class_label)
    return predicted_class_label

def generate_comment(person_count, class_dict):
    if person_count>0:
        comment = str(person_count) + " person(s) detected. "
        if len(class_dict)>0:
            comment = str(len(class_dict)) + " delivery person(s) detected. Distribution is " + str(class_dict)
    else:
        comment = "No Person Detetcted"
    return comment

def get_image_clasess(model, labels, image_path_list):
    class_dict = {}
    for path in image_path_list:
        class_ = get_image_class(model, labels, path)
        if class_ not in class_dict:
            class_dict[class_] = 1
        else:
            class_dict[class_]+=1
    return class_dict



# image_path = r"static\uploads\Uber eats Delivery Boy_17.jpg"
# print(get_image_class(image_path))

