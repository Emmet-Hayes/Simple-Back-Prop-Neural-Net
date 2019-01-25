#pragma once
#ifndef Neuron_H
#define Neuron_H
#include "Globalfuncs.h"
using namespace std;
struct Connection {           //each neuron has a connection structure
public:
	double weight;
	double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer; //for naming convenience

class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	vector<Connection> & getOutputWeights();
private:
	static double eta;        // [0.0..1.0] overall net training rate
	static double alpha;      // [0.0..n] multiplier of last weight change (momentum)
	static double transferFunction(double x); //transfer
	static double transferFunctionDerivative(double x);
	static double randomWeight() { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
};
#endif // !Neuron_H
