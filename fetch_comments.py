import re
import emoji
from langdetect import detect
from googleapiclient.discovery import build

def clean_comment (comment):
    comment = emoji.replace_emoji(comment, replace="")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()