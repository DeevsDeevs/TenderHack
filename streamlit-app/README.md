## Как запустить локально (Only Mac or Linux)

Для начала склонируйте репозиторий

```
git clone https://github.com/DeevsTheBest/TenderHack.git
cd TenderHack
```

Создайте окружения и загрузите библиотеки (python 3.9.6)\
[JamSpell](https://github.com/bakwc/JamSpell) может не устанавливаться. Issue по этому поводу [тут](https://github.com/bakwc/JamSpell/issues/73#issuecomment-465635464)

```
cd streamlit-app
python -m venv venv
venv/bin/activate
pip install gdown
pip install -r requirements.txt
```

Загрузите данные и веса через gdown (если не получается через него, то загрузите по прямым ссылкам и положите в данную директорию)
```python 3
gdown --fuzzy --folder 'https://drive.google.com/drive/folders/1JlgJluF50vJvq6QLbfmkQGXUgPOdFp7C?usp=sharing'
```

Для запуска локально
```
streamlit run app.py
```

Для запуска через ngrok (команды писать в разных консолях)
```
python server_ngrok.py
streamlit run --server.port 80 app.py
```
