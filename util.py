import pymysql
import hashlib

class User_Info:
    def __init__(self):
        conn = pymysql.connect(host='localhost',
                               user='root',
                               password='0000',
                               database='Smart_ATM')
        self.cursor = conn.cursor()

class Log_IN(User_Info):
    def __init__(self, cursor):
        super().__init__(cursor)
        self.ID = None
        self.PW = None
    
    def set_PW(self, PW):
        self.PW = PW
        
    def set_ID(self, ID):
        self.ID = ID
    
    def log_in_ID(self):
        self.cursor.execute('SELECT * FROM log_in WHERE user_id = %s', (self.ID,))
        user_info = self.cursor.fetchall()
        return bool(user_info)
    
    def log_in_PW(self):
        self.cursor.execute('SELECT encrypt_pw FROM user_info WHERE user_id = %s', (self.ID,))
        correct_PW = self.cursor.fetchone()
        
        if correct_PW and hashlib.sha256(self.PW.encode()).hexdigest() == correct_PW[0]:
            return True
            
        else:
            return False  

    def login_with_attempts(self):
        login_attempts = 0  # Initialize the login_attempts variable
        while login_attempts < 3:
            self.set_ID(input("Enter your username: "))
            self.set_PW(input("Enter your password: "))
            
            if self.log_in_ID() and self.log_in_PW():
                print("Login successful!")
                return True
            
            else:
                print("Login failed. Please try again.")
                login_attempts += 1
        else:
            print("Maximum login attempts reached. Account locked.")
            return False
 
    def sign_up(self, ID, NAME, PW):
        PW = hashlib.sha256(PW.encode()).hexdigest()
        self.cursor.execute('INSERT INTO user_info (user_id, name, encrypt_pw) VALUES (%s, %s, %s)', (ID, NAME, PW))
        self.cursor.connection.commit()
        self.cursor.close()

class Security(User_Info):
    def __init__(self):
        super().__init__()
