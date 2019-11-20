from pyspark.mllib.recommendation import ALS
import math

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

# build training data, validation data and testing data in 6, 2, 2(seed = 0(whatever...))
training_RDD, validation_RDD, test_RDD = complete_ratings_data.randomSplit([6, 2, 2], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

# train ALS model with training data and validation data
seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02
min_error = float('inf')
best_rank = -1
best_iteration = -1

for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank
print('The best model was trained with rank %s' % best_rank)

# get predictions results
predictions.take(3)
# get predictions results and real answer
rates_and_preds.take(3)

# use testing data to fit model
model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
print('For testing data the RMSE is %s' % (error))

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

# build a data
new_user_ID = 400
# The format of each line is (userID, movieID, rating)
new_user_ratings = [(400,1025,1),
                    (400,1161,5),
                    (400,1038,4)]   

    
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

# Save model
from pyspark.mllib.recommendation import MatrixFactorizationModel
model_path = "/home/cloudera/Desktop/datasets/music/movie_lens_als"
model.save(sc, model_path)

# load model
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
import math
model_path = "/home/cloudera/Desktop/datasets/music/movie_lens_als"
same_model = MatrixFactorizationModel.load(sc, model_path)