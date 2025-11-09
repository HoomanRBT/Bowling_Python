import mysql.connector


class MySQL:
    def __init__(self, host ='localhost', port='3307', user='root', passwd='%Hmd0914%', database='bowling-db', device_name=None):
        self.host = host
        self.user = user
        self.port = port
        self.passwd = passwd
        self.database = database
        self.device_name = device_name
        self.connection = mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            passwd=self.passwd,
            database=self.database
        )
        self.cursor = self.connection.cursor(prepared=True)
        
    print("hamed")