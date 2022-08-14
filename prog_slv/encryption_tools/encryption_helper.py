from Crypto.Cipher import AES
from base64 import b64encode, b64decode
import os

FILE_NAME = os.path.join(os.path.dirname(__file__), "encryption_txt")

class Crypt:
    def __init__(self, salt='SlTKeYOpHygTYkP3'):
        self.salt = salt.encode('utf8')
        self.enc_dec_method = 'utf-8'

    def make_key_from_password(self, password):
        assert len(password) <= 16
        key = password + "x" * (16 - len(password))
        return key

    def encrypt(self, str_to_enc, str_key):
        try:
            aes_obj = AES.new(str_key.encode('utf-8'), AES.MODE_CFB, self.salt)
            hx_enc = aes_obj.encrypt(str_to_enc.encode('utf8'))
            mret = b64encode(hx_enc).decode(self.enc_dec_method)
            return mret
        except ValueError as value_error:
            if value_error.args[0] == 'IV must be 16 bytes long':
                raise ValueError('Encryption Error: SALT must be 16 characters long')
            elif value_error.args[0] == 'AES key must be either 16, 24, or 32 bytes long':
                raise ValueError('Encryption Error: Encryption key must be either 16, 24, or 32 characters long')
            else:
                raise ValueError(value_error)

    def decrypt(self, enc_str, str_key):
        try:
            aes_obj = AES.new(str_key.encode('utf8'), AES.MODE_CFB, self.salt)
            str_tmp = b64decode(enc_str.encode(self.enc_dec_method))
            str_dec = aes_obj.decrypt(str_tmp)
            mret = str_dec.decode(self.enc_dec_method)
            return mret
        except ValueError as value_error:

            if value_error.args[0] == 'IV must be 16 bytes long':
                raise ValueError('Decryption Error: SALT must be 16 characters long')
            elif value_error.args[0] == 'AES key must be either 16, 24, or 32 bytes long':
                raise ValueError('Decryption Error: Encryption key must be either 16, 24, or 32 characters long')
            else:
                raise ValueError(value_error)

    def read_file(self):
        with open(FILE_NAME, 'r') as f:
            txt = f.read()
        return txt

    def read_password(self):
        password = input("Input key: ")
        str_key = self.make_key_from_password(password)
        return str_key

if __name__ == "__main__":
    crypt = Crypt()
    str_key = crypt.read_password()
    str_txt = crypt.read_file()

    #print(crypt.encrypt(str_txt, str_key))
    print(crypt.decrypt(str_txt, str_key))