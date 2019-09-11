import pymysql
import io

class DBConnection(object):
    def __init__(self):
        super(DBConnection, self).__init__()
        self.connection = pymysql.connect(host='localhost',
                                    user='root',
                                    password='123456',
                                    db='amazon_dataset',
                                    charset='utf8mb4',
                                    cursorclass=pymysql.cursors.DictCursor)
        self.sqlCmd = ""
        pass
    
    def Insertion(self,sql):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
            self.connection.commit()
            return True
        except Exception as ex:
            print('{}\n{}\n\n'.format(sql,str(ex)))
            with io.open(R"Insertion_Error.log", 'a',encoding='utf-8') as f:
                f.write('{}\n{}\n\n'.format(sql,str(ex)))
            return False
    
    def selection(self,sql):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                res = cursor.fetchall()
            self.connection.commit()
            return res
        except Exception as ex:
            print(ex)
    
    def close(self):
        self.connection.close()