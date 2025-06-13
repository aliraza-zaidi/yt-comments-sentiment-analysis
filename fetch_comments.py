import re
import emoji
from langdetect import detect
from googleapiclient.discovery import build

def clean_comment (comment):
    comment = emoji.replace_emoji(comment, replace="")
    comment = re.sub(r'[^\w\s]', '', comment)
    comment = re.sub(r'\s+', ' ', comment).strip()
    return comment.lower()

def is_english (comment):
    comment = comment.strip()
    if not comment or len(comment.split()) < 3:  # skip if too short
        return False
    try:
        return detect(comment) == 'en'
    except:
        return False

def fetch_comments(video_id, api_key, max_comments=100, max_len=150):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText',
            pageToken=next_page_token,
            order='relevance'
        )
        response = request.execute()
        
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            cleaned = clean_comment(comment)
            if is_english(cleaned) and len(cleaned) < max_len:
                comments.append(cleaned)
            if len(comments) >= max_comments:
                break

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments