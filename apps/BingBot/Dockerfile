FROM python:3.9.16
WORKDIR /bot
COPY requirements.txt /bot/
RUN pip install -r requirements.txt
COPY . /bot
CMD python bot.py
