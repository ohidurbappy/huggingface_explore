from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

image = Image.open("data/ohidur.jpg")

text = "Describe the image"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
print(model.config.id2label[outputs.logits.argmax(-1).item()])
# get second best answer
print(model.config.id2label[outputs.logits.argsort(-1, descending=True)[0, 1].item()])


# logits = outputs.logits
# idx = logits.argmax(-1).item()
# print("Predicted answer:", model.config.id2label[idx])



