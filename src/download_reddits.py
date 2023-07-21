import pandas as pd
import pandas_gbq
import google.auth


credentials, project = google.auth.default()

# Update the in-memory credentials cache (added in pandas-gbq 0.7.0).
pandas_gbq.context.credentials = credentials
print(project)

for y in ['2018', '2019']:
	for m in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
		ym = y+'_'+m

		quer = f"""SELECT s.subreddit as subreddit, 
		s.selftext as submission, a.body AS comment, b.body as reply, 
		s.score as submission_score, a.score as comment_score, b.score as reply_score, 
		s.author as submission_author, a.author as comment_author, b.author as reply_author
		FROM `fh-bigquery.reddit_comments.{ym}` a
		LEFT JOIN `fh-bigquery.reddit_comments.{ym}` b 
		ON CONCAT('t1_',a.id) = b.parent_id
		LEFT JOIN  `fh-bigquery.reddit_posts.{ym}` s
		ON CONCAT('t3_',s.id) = a.parent_id
		where b.body is not null 
		  and s.selftext is not null and s.selftext != ''
		  and b.author != s.author
		  and b.author != a.author
		  and s.subreddit IN ('writing', 'scifi', 'sciencefiction', 'MachineLearning', 'philosophy', 'cogsci', 'neuro', 'Futurology')
		"""
		
		tst = pd.read_gbq(quer,project_id=project,dialect='standard')
		tst.to_csv(f'./data/raw/myreddit_{ym}.csv')