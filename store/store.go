package store

import (
	"math/rand"
	"strings"
	"time"

	"github.com/mikemenaker/go-neural-network-service/model"
)

type NeuralNetwork interface {
	Train(cycles int)
	ForwardPass(x []float64) (sum float64)
	Trained() bool
}

var NeuralNetworks = make(map[string]NeuralNetwork)

func CreateNn(input [][]float64, actualOutput []float64) string {
	perceptron := model.NewPerceptron(input, actualOutput)

	id := randomString()
	NeuralNetworks[id] = &perceptron
	return id
}

func randomString() string {
	rand.Seed(time.Now().UnixNano())
	chars := []rune("ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
		"abcdefghijklmnopqrstuvwxyz" +
		"0123456789")
	length := 8
	var b strings.Builder
	for i := 0; i < length; i++ {
		b.WriteRune(chars[rand.Intn(len(chars))])
	}
	return b.String()
}
