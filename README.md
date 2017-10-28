# Keras in docker container:

## Create image:
```
docker build -t convetbasemodelc ./
```

## Start container with mapped volumes:
```
docker run -it -v /<path>/log:/script/models-logs/convnet/ convetbasemodelc
```
