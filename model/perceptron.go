package model

import (
	"math"
	"math/rand"
	"time"
)

// Perceptron is a neural network with n (# of features) input, 1 neuron, 1 output
type Perceptron struct {
	input        [][]float64
	actualOutput []float64
	weights      []float64
	bias         float64
	trained      bool
}

// NewPerceptron creates and initializes a new perceptron
func NewPerceptron(input [][]float64, actualOutput []float64) Perceptron {
	perceptron := Perceptron{
		input:        input,
		actualOutput: actualOutput,
		trained:      false,
	}
	perceptron.initialize()

	return perceptron
}

func (a *Perceptron) initialize() {
	rand.Seed(time.Now().UnixNano())
	a.bias = 0.0
	a.weights = make([]float64, len(a.input[0]))
	for i := 0; i < len(a.input[0]); i++ {
		a.weights[i] = rand.Float64()
	}
}

// Train the Perceptron for n cycles
func (a *Perceptron) Train(cycles int) {
	for i := 0; i < cycles; i++ {
		dw := make([]float64, len(a.input[0]))
		db := 0.0
		for length, val := range a.input {
			dw = vecAdd(dw, a.gradW(val, a.actualOutput[length]))
			db += a.gradB(val, a.actualOutput[length])
		}
		dw = scalarMatMul(2/float64(len(a.actualOutput)), dw)
		a.weights = vecAdd(a.weights, dw)
		a.bias += db * 2 / float64(len(a.actualOutput))
	}

	a.trained = true
}

// Trained returns if the perceptron has been trained
func (a *Perceptron) Trained() bool {
	return a.trained
}

// ForwardPass will do the prediction
func (a Perceptron) ForwardPass(x []float64) (sum float64) {
	return a.sigmoid(dotProduct(a.weights, x) + a.bias)
}

//Sigmoid Activation
func (a Perceptron) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (a Perceptron) gradW(x []float64, y float64) []float64 { //Calculate Gradients of Weights
	pred := a.ForwardPass(x)
	return scalarMatMul(-(pred-y)*pred*(1-pred), x)
}

func (a Perceptron) gradB(x []float64, y float64) float64 { //Calculate Gradients of Bias
	pred := a.ForwardPass(x)
	return -(pred - y) * pred * (1 - pred)
}

func dotProduct(v1, v2 []float64) float64 { //Dot Product of Two Vectors of same size
	dot := 0.0
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
	}
	return dot
}

func vecAdd(v1, v2 []float64) []float64 { //Addition of Two Vectors of same size
	add := make([]float64, len(v1))
	for i := 0; i < len(v1); i++ {
		add[i] = v1[i] + v2[i]
	}
	return add
}

func scalarMatMul(s float64, mat []float64) []float64 { //Multiplication of a Vector & Matrix
	result := make([]float64, len(mat))
	for i := 0; i < len(mat); i++ {
		result[i] += s * mat[i]
	}
	return result
}
