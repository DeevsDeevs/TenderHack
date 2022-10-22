from pyngrok import ngrok 
import os
os.system('ngrok authtoken 2ARsKtGKj47h7y4uXMQPrIeOinS_47Mkh6jkzNjFEJWuZYNEX')
url = ngrok.connect(port = 8501)
print(url)
input()