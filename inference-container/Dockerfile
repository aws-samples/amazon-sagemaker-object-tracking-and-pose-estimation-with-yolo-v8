FROM ultralytics/ultralytics:8.0.158-python
RUN pip install lap==0.4

##########################################################################################
# SageMaker requirements
##########################################################################################
## install flask
RUN pip install gevent
RUN pip install flask
RUN pip install gunicorn

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
 
COPY resources/*.* /app/
ENV WORKDIR /app/
WORKDIR /app/

ENTRYPOINT ["python", "serve.py"]
