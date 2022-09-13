# Overview :
The goal of this project is to use the BERT model to get the sentiment of youtube videos. This is done by using a pretrained twitter sent model and fine tuned for youtube video sentiment classification. 
# Getting the data :
We use the YouTube Data API v3 provided bu google to get an api key. We can then access the youtube comments by using the youtubve data API
# Usage :
```
python analyze.py -v video_link -n max_comments
```
**max_comments** is an argument to indicate maximum number of comments we want to analyze.