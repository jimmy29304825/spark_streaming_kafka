from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer


def output_rdd(rdd):
    # Transfer the rdd's data back to the Driver
    rdd_data = rdd.collect()

    # Create producer
    producer = KafkaProducer(bootstrap_servers=broker_list)
    # Get each (word,count) pair and send it to the topic by iterating the partition (an Iterable object)
    for word, count in rdd_data:
        message = "{},{}".format(word, str(count))
        producer.send(topic, value=bytes(message, "utf8"))

    producer.close()


if __name__ == "__main__":

    topic = "wordcount_result"
    broker_list = 'localhost:9092,localhost:9093'

    sc = SparkContext()
    ssc = StreamingContext(sc, 5)

    raw_stream = KafkaUtils.createStream(ssc, "localhost:2182", "consumer-group", {"test_stream": 3})
    raw_stream = ssc.socketTextStream("localhost", 9999)

    # Split each line into words
    words = raw_stream.flatMap(lambda line: str(line).split(" "))

    # Count each word in each batch
    pairs = words.map(lambda word: (word, 1))
    word_counts = pairs.reduceByKey(lambda x, y: x + y)


    word_counts.pprint()
    word_counts.foreachRDD(output_rdd)
    # Start it
    ssc.start()
    ssc.awaitTermination()
