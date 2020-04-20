# Go Neural Network Rest Service

Based on [Build Your Own Neural Network in Go](https://towardsdatascience.com/neural-network-from-scratch-in-go-language-b98e2abcced3)

## Build and Run
go build
go .\go-neural-network-service.exe

## Curl commands

### Create Network
```
curl --request POST \
  --url http://localhost:3030/neuralnetwork \
  --header 'cache-control: no-cache' \
  --header 'content-type: application/json' \
  --data '{\n	"input": [ [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0]],\n	"actualOutput": [0, 1, 1, 0]\n}'
```

### Train
```
  curl --request POST \
  --url http://localhost:3030/neuralnetwork/fNuPbAwZ/train \
  --header 'cache-control: no-cache' \
  --header 'content-type: application/json' \
  --data '{\n    "cycles": 1000\n}'
```
### Predict
```
  curl --request POST \
  --url http://localhost:3030/neuralnetwork/fNuPbAwZ/predict \
  --header 'cache-control: no-cache' \
  --header 'content-type: application/json' \
  --data '{\n	"input": [ 0, 1, 0 ]\n}'
  ```