from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
import math, time, json

# get message from rate_test
# send message to music_test
def test(rdd):
    if len(rdd.collect()) == 0:
        pass

    else:
        new_user_ratings = []
        
        new_user_ID = rdd.map(lambda x: eval(json.loads(x[1]))['userid']).collect()
        song_id = rdd.map(lambda x: eval(json.loads(x[1]))['songid']).collect()
        rating = rdd.map(lambda x: eval(json.loads(x[1]))['rating']).collect()
        
        new_user_list_len = len(rating)
        
        for i in range (new_user_list_len):
            data = (new_user_ID[i], song_id[i], rating[i])
            new_user_ratings.append(data)
        
        new_user_ratings_RDD = sc.parallelize(new_user_ratings)
        
        # merge new data into old data
        complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)
        
        # train model again with new data
        new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)
        
        try:
            new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just music IDs
        except:
            print(new_user_ratings,new_user_ratings_ids)
        
        # keep just those not on the ID list
        new_user_unrated_music_RDD = (complete_music_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID[0], x[0])))
        
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
        top_musics = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])
        new_user_ratings = []        
        return sc.parallelize(top_musics)

       

def output_rdd(rdd):
    # Transfer the rdd's data back to the Driver
    rdd_data = rdd.collect()

    # Create producer
    producer = KafkaProducer(bootstrap_servers=broker_list)
    # Get each (word,count) pair and send it to the topic by iterating the partition (an Iterable object)
    for count in rdd_data:
        message = "{}".format(str(count))
        producer.send(topic, value=bytes(message, "utf8"))

    producer.close()


if __name__ == "__main__":

    sc = SparkContext()
    ssc = StreamingContext(sc, 1)

    model_path = "/home/cloudera/Desktop/datasets/music/music_lens_als"
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
    complete_music_data = complete_music_raw_data.filter(lambda line: line!=complete_music_raw_data_header).map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2],tokens[3])).cache()
    complete_music_titles = complete_music_data.map(lambda x: (int(x[0]),x[1], x[2]))

    print("There are %s musics in the complete dataset" % (complete_music_titles.count()))   

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

    topic = "music_test"
    broker_list = 'localhost:9092,localhost:9093,localhost:9094'
    raw_stream = KafkaUtils.createStream(ssc, "localhost:2182", "consumer-group", {"rate_test": 3})

    data = raw_stream.transform(test)
    # data.pprint()
    data.foreachRDD(output_rdd)

    # Start it
    ssc.start()
    ssc.awaitTermination()