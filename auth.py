import pickle
import re
from typing import Optional
import os
from pathlib import Path

class AuthObject():

    def __init__(self) -> None:
        self.authpath = os.path.join(Path(__file__).parent, 'authobjectpickle.pkl')
        if not os.path.exists(self.authpath):
            config = {
                'credentials':{
                    'usernames':{
                        'admin':{
                            'email': 'admin@mail.in',
                            'name': 'admin',
                            'password': 'admin123',
                            'role': 'admin',
                            'state':1
                        },
                        'second_admin':{
                            'email':'secadmin@mail.in',
                            'name': 'secadmin',
                            'password': 'secadmin123',
                            'role':'translator',
                            'state':1
                        }
                    },
                    'cookie':{
                        'expiry_days': 1,
                        'key': 'mycookiekeyisveryrandom',
                        'name': 'authcookieadmin'
                    }
                }
            }
            nameList = [name for name in config['credentials']['usernames'].keys()]
            self.authobject = {
                'config': config,
                'nameList':nameList
            }

            with open('authobjectpickle.pkl', 'wb') as cookie:
                pickle.dump(obj=self.authobject, file=cookie)
        else:
            with open(self.authpath, 'rb') as authobj:
                self.authobject = pickle.load(authobj) 
            print("Default auth object loaded.")
        
    
    def get_auth(self):
        return self.authobject

    def adduser(self, username:str, email: Optional[str] = "", name: Optional[str] = "", password:str = "abc123", role: str = 'translator', status: int = 1):
        try:
            if username not in self.authobject['nameList']:
                if re.match(pattern='[\w*]@[\w*].[\w]', string=email):
                    self.authobject['config']['credentials']['usernames'][username] = {'email':email, 'name':name, 'password':password, 'role':role, 'status':status}
                    self.authobject['nameList'].append(username)
                    with open('authobjectpickle.pkl', 'wb') as cookie:
                        pickle.dump(obj=self.authobject, file=cookie)
                    return {'message':f"User {username} added to auth object"}
                else:
                    return {'message':f"Incorrect email format"}
            else:
                return {'message':f"User {username} already exists"}
        except Exception as e:
            return {"Error": f"{e}"}

    def delete_user(self, username:str):
        try:
            if username in self.authobject['nameList']:
                self.authobject['config']['credentials']['usernames'].pop(username)
                self.authobject['nameList'].pop(username)
                with open('authobjectpickle.pkl', 'wb') as cookie:
                    pickle.dump(obj=self.authobject, file=cookie)
                return {"object": username, 'message': f"{username} removed from auth object"}
            else:
                return {'message': f'User {username} does not exist'}
        except Exception as e:
            return {'Error': f"{e}"}
        
    def persist_auth_object(self):
        with open('authobjectpickle.pkl', 'wb') as cookie:
            pickle.dump(obj=self.authobject, file=cookie)
        return {'message':'Auth object updated'}
    
    def list_users(self):
        return


    



if __name__=="__main__":
    # Test code

    app_auth_object = AuthObject()
    print(app_auth_object.get_auth())

    print()

    add_status = app_auth_object.adduser(
        username='rahulkher',
        email='rahul.kher22gmail.com',
        name='rahul',
        password='R@#ulk#er123',
        role='sub-admin',
    )

    print(add_status['message'])

    print(app_auth_object.get_auth())