from transformers import AutoTokenizer, AutoModel , AutoConfig 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")

model = AutoModelForSequenceClassification.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")

tokenizer.save_pretrained("my_tokenizer.pth")
model.save_pretrained("my_model.pth")