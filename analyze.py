from obsei.source import YoutubeScrapperSource, YoutubeScrapperConfig
import argparse
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas
import matplotlib.pyplot as plt


def get_data(video_url):
    source_config = YoutubeScrapperConfig(
    video_url= video_url,
    fetch_replies=False,
    max_comments=20,
    lookup_period="1Y",
    )
    source = YoutubeScrapperSource()
    source_response_list = source.lookup(source_config)
    
    comments = []
    likes = []
    for idx, source_response in enumerate(source_response_list):
        comments.append(source_response.__dict__['meta']['text'])
        likes.append(source_response.__dict__['meta']['votes'])
    
    return comments, likes

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def load_model():
    
    # Tasks:
    # emoji, emotion, hate, irony, offensive, sentiment
    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # download label mapping
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    return model, labels, tokenizer
    
def vid_score(labels,model, df,tokenizer):
    label_length = len(labels)
    print(label_length)
    vid_score = [0 for i in range(label_length)]
    comment_len = len(df.loc[:, "comments"])
    for comment in df.loc[:, "comments"]:
        comment = preprocess(comment)
        encoded_input = tokenizer(comment, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        vid_score[i] += s
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
    vid_score = [e/comment_len for e in vid_score]
    return vid_score
    
def show(labels, values):
    
    
    plt.rcParams['figure.figsize'] = [20, 10]
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
    
def main():
    parser = argparse.ArgumentParser("This file gives sentiment analysis of youtube comments")
    parser.add_argument("-v",help = "Youtube video link")
    args = parser.parse_args()
    v = args.v
    comments,likes = get_data(v)
    df = pandas.DataFrame(columns = ["comments","likes"])
    df["comments"] = comments
    df["likes"] = likes
    model, labels, tokenizer = load_model()
    score = vid_score(labels,model, df,tokenizer)
    show(labels, score)
    
if __name__ == "__main__":
    main()