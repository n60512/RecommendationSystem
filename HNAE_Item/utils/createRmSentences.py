from DBconnector import DBConnection
from nltk.corpus import stopwords
import nltk
import tqdm


if __name__ == "__main__":
    sql = "select `ID`, reviewText FROM toys_interaction6;"

    conn = DBConnection()
    res = conn.selection(sql)
    
    for row in tqdm.tqdm(res):
        uid = row['ID']
        reviewText = row['reviewText']
        filtered_words = [word for word in reviewText.split(' ') if word not in stopwords.words('english')]
        filtered_sentences = " ".join(filtered_words)

        # print(reviewText)
        # print(filtered_sentences)

        insertsql = ("INSERT INTO toys_interaction6_rm_sw " + 
                    "(`ID`, reviewText) VALUES ({},'{}');".format(uid, filtered_sentences)
                    )
        conn.Insertion(insertsql)

        stop = 1

    conn.close()


    


