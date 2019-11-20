# load model
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
import math, time
model_path = "/home/cloudera/Desktop/datasets/music/movie_lens_als"
same_model = MatrixFactorizationModel.load(sc, model_path)
# load data
complete_ratings_raw_data = sc.textFile("/home/cloudera/Desktop/datasets/music/ff2_data.csv")
# set column names
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]
# transfer data
## 1. filiter: remove first row
## 2. map: split data by ","
## 3. map: get first 3 columns
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header).map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
print("There are %s recommendations in the complete dataset" % (complete_ratings_data.count()))

# train ALS model with training data and validation data
seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02
min_error = float('inf')
best_rank = 4
best_iteration = -1

# load music neta data
complete_music_raw_data = sc.textFile("/home/cloudera/Desktop/datasets/music/songs_metadata_file_new.csv")
complete_music_raw_data_header = complete_music_raw_data.take(1)[0]
complete_music_data = complete_music_raw_data.filter(lambda line: line!=complete_music_raw_data_header).map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
complete_music_titles = complete_music_data.map(lambda x: (int(x[0]),x[1]))
print("There are %s movies in the complete dataset" % (complete_music_titles.count()))

# join data's def
def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

# group by every mucis couts data by music id
music_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
# get average and counts for each music 
music_ID_with_avg_ratings_RDD = music_ID_with_ratings_RDD.map(get_counts_and_averages)
# get counts for each music 
music_rating_counts_RDD = music_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

    
# 設定要連線到Kafka集群的相關設定, 產生一個Kafka的Consumer的實例
consumer = KafkaConsumer(
    # 指定Kafka集群伺服器
    bootstrap_servers=["kafka:9092"],
    value_deserializer=lambda m: json.loads(m.decode('ascii')),
    auto_offset_reset="earliest"
)

# 讓Consumer向Kafka集群訂閱指定的topic
consumer.subscribe(topics="music")

# 持續的拉取Kafka有進來的訊息
print("Now listening for incoming messages ...")
# 持續監控是否有新的record進來
for record in consumer:
    topic = record.topic
    partition = record.partition
    offset = record.offset
    timestamp = record.timestamp
    # 取出msgKey與msgValue
    msgKey = record.key
    msgValue = record.value
    # 秀出metadata與msgKey & msgValue訊息
    # {'userid': 200, 'music_rate1': (1111, 5), 'music_rate2': (1452, 1), 'music_rate3': (1354, 3)}
    print("topic=%s, partition=%s, offset=%s : (key=%s, value=%s)" % (record.topic, record.partition, record.offset, record.key, record.value))# transfer data to RDD
    new_user_ID = int(msgValue["userid"])
    new_user_ratings = [(new_user_ID,int(list(msgValue["music_rate1"])[0]),int(list(msgValue["music_rate1"])[1]),
                        (new_user_ID,int(list(msgValue["music_rate2"])[0]),int(list(msgValue["music_rate2"])[1]),
                        (new_user_ID,int(list(msgValue["music_rate3"])[0]),int(list(msgValue["music_rate3"])[1])]
    print(new_user_ratings)
    new_user_ratings_RDD = sc.parallelize(new_user_ratings)
    print('New user ratings: %s' % new_user_ratings_RDD.take(3))
    # merge new data into old data
    complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)
    # train model again with new data
    from time import time
    t0 = time()
    new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)
    tt = time() - t0
    print("New model trained in %s seconds" % round(tt,3))

    new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just movie IDs
    # keep just those not on the ID list
    new_user_unrated_music_RDD = (complete_music_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))
    # Use the input RDD, new_user_unrated_music_RDD, with new_ratings_model.predictAll() to predict new ratings for the musics
    new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_music_RDD)

    # get every predicct result for new user
    new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
    # merge data with music info
    new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_RDD.join(complete_music_titles).join(music_rating_counts_RDD)
    new_user_recommendations_rating_title_and_count_RDD.take(3)
    # transfer data format
    new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
    # sort data by rating score and list first 25 data
    top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])
    print('TOP recommended movies (with more than 25 reviews):\n%s' % '\n'.join(map(str, top_movies)))