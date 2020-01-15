from utils.DBconnector import DBConnection
from utils.preprocessing import Preprocess
from nltk.corpus import stopwords
import tqdm

# Generating sentences that remove stopwords by nltk.
def generate_rm_sw_sentences(conn):
    
    pre_work = Preprocess()
    sql = "SELECT `ID`, reviewText FROM clothing_interaction6;"
    res = conn.selection(sql)

    for index in tqdm.tqdm(range(len(res))):
        
        id = res[index]['ID']
        reviewText = res[index]['reviewText']

        nor_sentence = pre_work.normalizeString(reviewText)
        
        sentence_segment = nor_sentence.split(' ')
        filtered_sentence = [word for word in sentence_segment if word not in stopwords.words('english')]

        filtered_sentence = " ".join(filtered_sentence)

        insert_sql = "INSERT INTO clothing_interaction6_rm_sw (`ID`, reviewText) VALUES ('{}', '{}');".format(id, filtered_sentence)
        conn.Insertion(insert_sql)  


# Generating sentences length 
def generate_sentences_length(conn):

    sql = "SELECT `ID`, reviewText FROM clothing_interaction6_rm_sw;"

    res = conn.selection(sql)

    for index in tqdm.tqdm(range(len(res))):
        
        id = res[index]['ID']
        reviewText = res[index]['reviewText']
        
        sentence_segment = reviewText.split(' ')
        sentence_length = len(sentence_segment)
            
        insert_sql = "INSERT INTO clothing_interaction6_sen_len (`ID`, sentence_length) VALUES ('{}', {});".format(id, sentence_length)
        conn.Insertion(insert_sql)


conn = DBConnection()
# generate_rm_sw_sentences(conn)
# generate_sentences_length(conn)
conn.close()