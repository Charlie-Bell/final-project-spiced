import requests
from requests.auth import HTTPBasicAuth
import praw
import sys
from  pprint import pprint
from dotenv import load_dotenv
from os import getenv
import pymongo


sys.stdout.reconfigure(encoding='utf-8') # Useful for windows user

load_dotenv()

# Authenticate
reddit = praw.Reddit(
    client_id=getenv('client_id'),
    client_secret=getenv('secret'),
    password=getenv('password'),
    username=getenv('username'),
    user_agent="Agent"
)

# Choose Sub-Reddit
subreddit = reddit.subreddit("LifeProTips")

# Create Mongo Client
client = pymongo.MongoClient(host="mongodb", port=27017)
db = client.reddits

# Extraction loop
def extraction_loop(time_filter="all", limit=500):
    for submission in subreddit.top(time_filter=time_filter, limit=limit):
        title = submission.title
        upvotes = submission.score
        _id = submission.id

        mongo_input = {'_id':_id, 'title': title, 'upvotes': upvotes}
        try: 
            db.posts.insert_one(dict(mongo_input))
        except pymongo.errors.DuplicateKeyError:
            continue

#extraction_loop(time_filter="all", limit=500)
extraction_loop(time_filter="year", limit=10000)