import pymongo
from sqlalchemy import create_engine, text
from time import sleep


sleep(1)

# Establish a connection to the MongoDB server
client = pymongo.MongoClient(host="mongodb", port=27017)

# Select the database you want to use withing the MongoDB server
db = client.reddits

# Get credentials for Postgres server
username = "postgres"
password = "postgres"
postgres_db = "postgresdb"
postgres_host = "postgresdb"

pg_client = create_engine(f"postgresql://{username}:{password}@{postgres_host}:5432/{postgres_db}", echo=True)
# Connect the client to postgres
pg_client_connect = pg_client.connect()

# Delete Postgres table 'reddits'
create_table = text(
   """
      DROP TABLE IF EXISTS reddits;
   """)
# Execute the query create_table
pg_client_connect.execute(create_table)
pg_client_connect.commit()

# Create Postgres table 'reddits'
create_table = text(
   """
      CREATE TABLE IF NOT EXISTS reddits (
      title VARCHAR(500),
      upvotes NUMERIC
   );
   """)
# Execute the query create_table
pg_client_connect.execute(create_table)
pg_client_connect.commit()

sleep(3)

# Copy data from MongoDB to Postgres
docs = list(db.posts.find())
for doc in docs:  
    # Insert data
    title = doc['title'].replace("'", " ") # for cleaning the text see also the below *Debugging hints*
    upvotes = doc['upvotes']
    print(title, upvotes)

    insert = text(f"INSERT INTO reddits VALUES ('{title}', {upvotes});")
    # Execute the query insert
    pg_client_connect.execute(insert)
    pg_client_connect.commit()