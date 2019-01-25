#include "Neuron.h"
double Neuron::eta = 0.33;                            // static private member for learning rate, [0.0..1.0]
double Neuron::alpha = 0.55;                          // momentum, multiplier of last deltaWeight, [0.0..1.0]

/* Neurons initialize new connections, with a random weight values, 
   the index of the neuron is passed and set. */
Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
	for (unsigned c = 0; c < numOutputs; ++c) { 
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}

/* the previous layers will help us determine the new weights.*/
void Neuron::updateInputWeights(Layer &prevLayer) {
	for (unsigned n = 0; n < prevLayer.size(); ++n) { // The weights to be updated are in the Connection container in the neurons of the previous layer
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		double newDeltaWeight =
			eta                                       // Individual input, magnified by the gradient and train rate:
			* neuron.getOutputVal()
			* m_gradient
			+ alpha                                   // Alpha how much of the previous delta weight to consider
			* oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight; //this is the derivative in action
	}
}

/* returns the */
vector<Connection> & Neuron::getOutputWeights() {
	return m_outputWeights;
}

/* sums our contributions of the errors at the nodes we feed */
double Neuron::sumDOW(const Layer &nextLayer) const {
	double sum = 0.0; 
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) 
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	return sum;
}

/* calculates the new gradient for a hidden layer of neurons */
void Neuron::calcHiddenGradients(const Layer &nextLayer) {
	double dow = sumDOW(nextLayer);
	m_gradient = dow * transferFunctionDerivative(m_outputVal);
}

/* calculates the new gradient for the output layer. */
void Neuron::calcOutputGradients(double targetVal) {
	double delta = targetVal - m_outputVal;
	m_gradient = delta * transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x) {
	return tanh(x);                                  // tanh - output range [-1.0..1.0]
}

double Neuron::transferFunctionDerivative(double x) {
	return 1.0 - x * x;                              // tanh derivative
}

void Neuron::feedForward(const Layer &prevLayer) {
	double sum = 0.0;
	                         // Sum the previous layer's outputs (which are our inputs)
	                         //Includes the bias node from the previous layer.
	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	m_outputVal = transferFunction(sum);
}