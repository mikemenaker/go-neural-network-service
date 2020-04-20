package route

import (
	"net/http"

	"github.com/mikemenaker/go-neural-network-service/controller"
)

// Route defines a route into the service
type Route struct {
	Name        string
	Method      string
	Pattern     string
	HandlerFunc http.HandlerFunc
}

// Routes is all the routes need to create/train/predict a nueral network
type Routes []Route

var routes = Routes{
	Route{
		"Create",
		"POST",
		"/neuralnetwork",
		controller.Create,
	},
	Route{
		"Train",
		"POST",
		"/neuralnetwork/{id}/train",
		controller.Train,
	},
	Route{
		"Predict",
		"POST",
		"/neuralnetwork/{id}/predict",
		controller.Predict,
	},
	// Route{
	// 	"Delete",
	// 	"DELETE",
	// 	"/neuralnetworks/{id:[0-9]+}",
	// 	Secure,
	// },
}
