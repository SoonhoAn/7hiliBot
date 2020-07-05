import torch
import logging

# !pip install transformers==2.8.0
from transformers import AutoModelWithLMHead, AutoTokenizer

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")
    model.to(device)
    model.eval()
    return model, tokenizer
