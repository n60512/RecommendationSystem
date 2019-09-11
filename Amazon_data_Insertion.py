from DBconnector import DBConnection
import json
import gzip
import re

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

def RegularExpression(rule,original_text):
    return " ".join(re.findall(rule, original_text))


def ReviewInsertion():
    DBconn = DBConnection() 
    counter = 1
    print_every = 1000
    
    for l in parse(R"F:\Dataset\Amazon\Electonics\reviews_Electronics_5.json.gz"):  
        if counter % print_every == 0:
            print('[{}] rows data has done.'.format(counter))
        counter += 1

        data = (json.loads(l))
        helpfulRating = lambda m,n : n if n == 0 else (m/n)
      
        # 處理名字 (空)
        hasObject = lambda obj : data[obj] if obj in data else None
        hasStr = lambda text : ("\"{}\"").format(RegularExpression('[a-zA-Z0-9.]+', text)) if text is not None else 'NULL'

        reviewerID : str = data['reviewerID']
        asin : str = data['asin']
        reviewerName : str = hasStr(hasObject('reviewerName'))
        helpful : float = helpfulRating( data['helpful'][0], data['helpful'][1] )
        reviewText : str = RegularExpression('[a-zA-Z0-9]+', data['reviewText'])
        overall : float = data['overall']
        summary : str = RegularExpression('[a-zA-Z0-9]+', data['summary'])
        unixReviewTime : str = data['unixReviewTime']
        reviewTime : str = data['reviewTime']    
      
        insertSql = ("INSERT INTO `review` ( \n"+
                    "`reviewerID`, \n"+
                    "`asin`, \n"+
                    "`reviewerName`, \n"+
                    "`helpful`, \n"+
                    "`reviewText`, \n"+
                    "`overall`, \n"+
                    "`summary`, \n"+
                    "`unixReviewTime`, \n"+
                    "`reviewTime` ) \n"+
                    "VALUES \n"+
                    "( \"{}\",\"{}\",{},{},\"{}\",{},\"{}\",\"{}\",\"{}\");".
                    format(reviewerID, asin, reviewerName, helpful, 
                    reviewText, overall, summary, unixReviewTime, reviewTime)
                    )
        DBconn.Insertion(insertSql)
    DBconn.connection.close()

def metadataInsertion():
    DBconn = DBConnection() 
    counter = 1
    print_every = 1000
    
    for l in parse(R"F:\Dataset\Amazon\Electonics\meta_Electronics.json.gz"):  
        if counter % print_every == 0:
            print('[{}] rows data has done.'.format(counter))
        counter += 1

        data = (json.loads(l))

        # 處理名字 (空)
        hasObject = lambda obj : data[obj] if obj in data else None
        hasFloat = lambda obj : data[obj] if obj in data else 'NULL'
        hasStr = lambda text : ("\"{}\"").format(RegularExpression('[a-zA-Z0-9,.]+', text)) if text is not None else 'NULL'

        asin : str = data['asin']
        imUrl : str = hasObject('imUrl')
        brand : str = hasObject('brand')
        description : str = hasStr(hasObject('description'))
        title : str = hasStr(hasObject('title'))
        price : float = hasFloat('price')
        related : str = hasObject('related')
        categories : str = hasObject('categories')
        
        # print('title:\t{}\nbrand:\t{}\nprice:\t{}\nrelated:\t{}\ncategories:\t{}\n============================'.
        #     format(title,brand,price,related,categories))
        
        insertSql = ("INSERT INTO `metadata` ( \n"+
                    "`asin`, \n"+ 
                    "`title`, \n"+ 
                    "`description`, \n"+ 
                    "`price`, \n"+ 
                    "`imUrl`, \n"+ 
                    "`related`, \n"+ 
                    "`categories`) \n"+
                    "VALUES (\"{}\",{},{},{},\"{}\",\"{}\",\"{}\");".
                    format(asin,title,description,price,imUrl,related,categories))
        
        print(insertSql)
        DBconn.Insertion(insertSql)
    DBconn.connection.close()    

if __name__ == "__main__":
    metadataInsertion()
    pass