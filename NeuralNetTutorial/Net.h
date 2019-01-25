#pragma once
#ifndef Net_H
#define Net_H
#include "Neuron.h"
class Net {
public:
	Net(const vector<unsigned> &);
	void feedForward(const vector<double> &);
	void backProp(const vector<double> &);
	void getResults(vector<double> &) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }
	void writeNet(const vector<unsigned> &);
	void readNet(vector<unsigned> &);
private:
	vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};
#endif