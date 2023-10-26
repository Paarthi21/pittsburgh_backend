FROM python:3.8.0

RUN pip install --upgrade pip

RUN apt-get update -qq 

RUN apt-get install -y -qq libtesseract-dev libleptonica-dev tesseract-ocr

RUN apt-get install -y -qq \
  tesseract-ocr-eng \
  tesseract-ocr-ara \
  tesseract-ocr-script-deva


RUN apt-get install -y poppler-utils 

WORKDIR /app

ENV FLASK_APP=app.py 

ENV FLASK_ENV=development

COPY ./requirements.txt .


RUN pip install -r requirements.txt

# RUN python -m nltk.downloader punkt
#RUN python -m nltk.downloader stopwords
#RUN python -m nltk.downloader wordnet
RUN pip install nltk
RUN python -c "import nltk; nltk.download('punkt')"

RUN python -c "import nltk; nltk.download('stopwords')"

RUN python -c "import nltk; nltk.download('wordnet')"

EXPOSE 5000

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0","--port","5000"]

