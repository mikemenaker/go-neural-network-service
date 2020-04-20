package controller

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/mikemenaker/go-neural-network-service/store"
)

type createPayload struct {
	Input        [][]float64 `json:"input"`
	ActualOutput []float64   `json:"actualOutput"`
}

type trainPayload struct {
	Cycles int `json:"Cycles"`
}

type predictPayload struct {
	Input []float64 `json:"input"`
}

type createResponse struct {
	ID string `json:"id"`
}

type errorResponse struct {
	Message string `json:"message"`
}

type predictResponse struct {
	Prediction float64 `json:"prediction"`
}

// Create will create a new neural network and return the id
func Create(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(r.Body)
	var data createPayload
	err := decoder.Decode(&data)
	if err != nil {
		panic(err)
	}

	id := store.CreateNn(data.Input, data.ActualOutput)
	log.Printf("Created NN, id = %s", id)

	sendResponse(w, createResponse{ID: id}, http.StatusOK)
}

// Train will train a created nueral network with {Cycles} times
func Train(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(r.Body)
	var data trainPayload
	err := decoder.Decode(&data)
	if err != nil {
		panic(err)
	}

	params := mux.Vars(r)
	id := params["id"]
	log.Printf("Id = %s, Cycles = %d", id, data.Cycles)

	nn, ok := store.NeuralNetworks[id]
	log.Printf("Training NN: %+v\n", nn)

	nn.Train(data.Cycles)
	log.Printf("Trained NN: %+v\n", nn)
	store.NeuralNetworks[id] = nn

	if ok {
		sendResponse(w, createResponse{ID: id}, http.StatusOK)
	} else {
		sendResponse(w, errorResponse{Message: "Doesn't exist"}, 404)
	}
}

// Predict will take an input and make a prediction on a trained neural network
func Predict(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(r.Body)
	var data predictPayload
	err := decoder.Decode(&data)
	if err != nil {
		panic(err)
	}
	params := mux.Vars(r)
	id := params["id"]
	log.Printf("Id = %s, Input = %+v\n", id, data.Input)

	nn, ok := store.NeuralNetworks[id]
	prediction := nn.ForwardPass(data.Input)
	if ok {
		if nn.Trained() {
			sendResponse(w, predictResponse{Prediction: prediction}, http.StatusOK)
		} else {
			sendResponse(w, errorResponse{Message: "Not trained"}, 400)
		}
	} else {
		sendResponse(w, errorResponse{Message: "Doesn't exist"}, 404)
	}
}

func sendResponse(w http.ResponseWriter, data interface{}, statusCode int) {
	w.Header().Set("Content-Type", "application/json; charset=UTF-8")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		panic(err)
	}
}
