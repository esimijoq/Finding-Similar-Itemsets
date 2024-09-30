!pip install pyspark
!pip install -U -q PyDrive
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
import hashlib
import pandas as pd
import numpy as np
from collections import defaultdict

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

id='1q-0N3l5O1d_RBrC_c-UlxmABH-xIVBH5'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('tactic_evals.csv')


spark = SparkSession.builder.appName("DocumentSimilarity").getOrCreate()

df = spark.read.csv('tactic_evals.csv', header=True, inferSchema=True)
df = df.withColumnRenamed("FEN", "text")
total = 250
df = df.limit(total)

#Step 1: Shingling: Converts a document into a set
def k_shingles(text, k):
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i + k]
        shingle_hash = hashlib.md5(shingle.encode('utf-8')).hexdigest()[:8]
        shingles.add(shingle_hash)
    return list(shingles)

k = 16
k_shingles_udf = udf(lambda text: k_shingles(text, k), ArrayType(StringType()))
df_shingled = df.withColumn("shingles", k_shingles_udf(df["text"]))

exploded_df = df_shingled.select("text", explode("shingles").alias("shingle"))
unique_shingles_df = exploded_df.select("shingle").distinct()
unique_shingles = unique_shingles_df.rdd.flatMap(lambda x: x).collect()
shingle_index = {shingle: idx for idx, shingle in enumerate(unique_shingles)}

def encode_shingles(shingles, shingle_index):
    vector = np.zeros(len(shingle_index), dtype=int)
    for shingle in shingles:
        if shingle in shingle_index:
            vector[shingle_index[shingle]] = 1
    return vector.tolist()

encode_shingles_udf = udf(lambda shingles: encode_shingles(shingles, shingle_index), ArrayType(IntegerType()))
df_encoded = df_shingled.withColumn("encoded_vector", encode_shingles_udf(df_shingled["shingles"]))

pdf_encoded = df_encoded.select("encoded_vector").toPandas()
matrix = np.array(pdf_encoded["encoded_vector"].tolist())


#Step 2 Minhashing(creating signatures)
def generate_permutations(num_rows, num_permutations):
    return [np.random.permutation(num_rows) for _ in range(num_permutations)]

def minhash_signature(matrix, permutation):
    num_cols = matrix.shape[1]
    signature = np.full(num_cols, np.inf)
    for col in range(num_cols):
        for row_idx in permutation:
            if matrix[row_idx, col] == 1:
                signature[col] = row_idx
                break
    return signature

def compute_minhash_signatures(matrix, permutations):
    num_permutations = len(permutations)
    num_cols = matrix.shape[1]
    signatures = np.zeros((num_permutations, num_cols))

    for i, perm in enumerate(permutations):
        signatures[i] = minhash_signature(matrix, perm)

    return signatures

num_rows, num_cols = matrix.shape
num_permutations = 100
permutations = generate_permutations(num_rows, num_permutations)

signatures = compute_minhash_signatures(matrix, permutations)

def split_signature_matrix(signatures, b):
    num_rows, num_cols = signatures.shape
    r = num_rows // b
    bands = [signatures[i*r:(i+1)*r, :] for i in range(b)]
    return bands, r

#Step 3 LSH (candidate pairs in the same buckets)
def hash_band_content(content, num_buckets):
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return int(content_hash, 16) % num_buckets

def hash_band(band, num_buckets):
    hash_table = defaultdict(set)
    for doc_id in range(band.shape[1]):
        band_content = ''.join(map(str, band[:, doc_id]))
        bucket_id = hash_band_content(band_content, num_buckets)
        hash_table[bucket_id].add(doc_id)
    return hash_table

def hash_bands(bands, num_buckets):
    all_hash_tables = []
    for band in bands:
        hash_table = hash_band(band, num_buckets)
        all_hash_tables.append(hash_table)
    return all_hash_tables

def find_candidate_pairs_from_hash_tables(hash_tables):
    candidate_pairs = set()
    for hash_table in hash_tables:
        for bucket_id, doc_ids in hash_table.items():
            doc_ids = list(doc_ids)
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    candidate_pairs.add((doc_ids[i], doc_ids[j]))
    return candidate_pairs

b = 50
num_buckets = 20
bands, r = split_signature_matrix(signatures, b)

hash_tables = hash_bands(bands, num_buckets)

candidate_pairs = find_candidate_pairs_from_hash_tables(hash_tables)

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def filter_similar_pairs(candidate_pairs, shingles_df, threshold):
    similar_pairs = []
    shingles_list = shingles_df.rdd.map(lambda row: row["shingles"]).collect()
    shingles_dict = {i: set(shingles) for i, shingles in enumerate(shingles_list)}
    for (i, j) in candidate_pairs:
        if i < len(shingles_dict) and j < len(shingles_dict):
            set_i = shingles_dict[i]
            set_j = shingles_dict[j]
            similarity = jaccard_similarity(set_i, set_j)
            if similarity >= threshold:
                similar_pairs.append((i, j, similarity))
    return similar_pairs

similarity_threshold = 0.5
similar_pairs = filter_similar_pairs(candidate_pairs, df_shingled, similarity_threshold)

def create_similarity_df(pairs, df):
    data = []
    for i, j, sim in pairs:
        text1 = df.collect()[i]["text"]
        text2 = df.collect()[j]["text"]
        data.append((i, j, sim, text1, text2))
    if not data:
        return None
    similarity_df = spark.createDataFrame(data, ["doc1", "doc2", "similarity", "doc1_text", "doc2_text"])
    return similarity_df

similarity_df = create_similarity_df(similar_pairs, df)
if similarity_df:
    similarity_df.show(truncate=False)

def create_similarity_dataframe(pairs):
    df = pd.DataFrame(pairs, columns=['Doc 1', 'Doc 2', 'Similarity'])
    return df

spark.stop()
