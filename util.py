import pymysql
import hashlib

class User_Info:
    def __init__(self):
        conn = pymysql.connect(host = 'localhost',
                               user = 'root',
                               password = '0000',
                               database = 'Smart_ATM')
        self.cursor = conn.cursor()

class Log_IN(User_Info):
    super().__init__()
    
    def check_id(self,ID):
        self.cursor.execute('select * from log_in where user_id = \'{}\''.format(ID))
        return self.cursor.fetchall()
        
    def log_in_ID(self,ID):
        user_info = self.check_id(ID)
        if user_info:
            True
        else:
            False
    
    def log_in_PW(self,ID,PW):
        correct_PW = self.cursor.execute('''
                                         SELECT encrypt_pw
                                         FROM user_info
                                         WHERE user_id = '{}';
                                         '''.format(ID))
        
        if hashlib.sha256(PW.encode()).hexdigest() == correct_PW:
            return True
            
        else:
            return False
            
    def sign_up(self, ID, NAME,PW):
        PW = hashlib.sha256(PW.encode()).hexdigest()
        self.cursor.execute('INSERT INTO user_info (user_id, name, encrypt_pw) VALUES (\'{}\', \'{}}\', \'{}\');'.format(ID,NAME,PW))
        self.cursor.commit()
        self.cursor.close()

    
class Security(User_Info):
    def __init__(self):
        super().__init__()
    
    
        
        
        