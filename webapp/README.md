## Preparation
```
$ pip install -r requirements.txt
```

## Execute app locally
```
$ uvicorn app:app --host 127.0.0.1 --port 8080
```

## Test app loaclly
```
$ curl -X POST "http://127.0.0.1:8080/predict/image" \
  -F "file=@test.jpg" \
  -o out.png

```

## Build and rundocker container
```
$ docker build -t image-api .
$ docker run -p 8080:8080 image-api
```

## GCP server deployment
```
$ gcloud run deploy chenfeng-image-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --concurrency 1

```
