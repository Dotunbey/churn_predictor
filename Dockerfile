FROM python:3.11-slim

RUN pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=1.0.bin", "./"]
EXPOSE 9696
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]

