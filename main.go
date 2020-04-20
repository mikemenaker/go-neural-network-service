package main

import (
	"log"
	"net/http"

	"github.com/mikemenaker/go-neural-network-service/route"
)

func main() {
	router := route.NewRouter()
	log.Fatal(http.ListenAndServe(":3030", router))
}
