#include "Net.h"
double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

/* fills the network with layers of neurons, with a bias neuron in each layer. */
Net::Net(const vector<unsigned> &topology) {
	for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) { // We have a new layer, now fill it with neurons, and
			m_layers.back().push_back(Neuron(numOutputs, neuronNum)); 	             // adds a bias neuron in each layer.
		}
		m_layers.back().back().setOutputVal(1.0); // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
	}
}

/* fills the passed vector of values with results from the output values. */
void Net::getResults(vector<double> &resultVals) const {
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());  //fill result vector
	}
}

/**/
void Net::backProp(const vector<double> &targetVals) {
	Layer &outputLayer = m_layers.back(); // Calculate overall net error (RMS of output neuron errors)
	m_error = 0.0;
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1;    // get average error squared
	m_error = sqrt(m_error);              // RMS
	m_recentAverageError = 	              // Implement a recent average measurement
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}
	// Calculate hidden layer gradients
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];
		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}
	// For all layers from outputs to first hidden layer,
	// update connection weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const vector<double> &inputVals) {
	assert(inputVals.size() == m_layers[0].size() - 1);
	// Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::writeNet(const vector<unsigned> & topology) {
	//make sure every layer is accurate with the topology.
	for (unsigned c = 0; c < topology.size(); ++c) assert(m_layers.at(c).size() - 1 == topology.at(c));
	ofstream outFile("learnDataWeights.txt");
	outFile.seekp(0, ios::beg);
	for (unsigned i = 0; i < topology.size(); ++i) {
		for (unsigned j = 0; j < topology.at(i); ++j) {
			for (unsigned k = 0; k < m_layers.at(i).at(j).getOutputWeights().size(); ++k) {
				outFile << "W: " << m_layers.at(i).at(j).getOutputWeights().at(k).weight << "\n"
					<< "DW: " << m_layers.at(i).at(j).getOutputWeights().at(k).deltaWeight << "\n";
			}
		}
	}
	outFile.close();
}

void Net::readNet(vector<unsigned> & topology) {
	for (unsigned c = 0; c < topology.size(); ++c) assert(this->m_layers.at(c).size() - 1 == topology.at(c));
	vector<double> tWeights, tDeltaWeights;
	ifstream inFile("learnDataWeights.txt");
	unsigned lCount = 0;
	while (!inFile.eof()) {
		string line;
		getline(inFile, line);
		stringstream ss(line);
		string label;
		ss >> label;
		if (label.compare("W:") == 0) {
			double oneValue;
			while (ss >> oneValue) {
				tWeights.push_back(oneValue);
			}
		}
		string line2;
		getline(inFile, line2);
		stringstream ss2(line2);
		string label2;
		ss2 >> label2;
		if (label2.compare("DW:") == 0) {
			double oneValue2;
			while (ss2 >> oneValue2) {
				tDeltaWeights.push_back(oneValue2);
			}
		}
		++lCount;
	}
	assert(tWeights.size() == tDeltaWeights.size());
	unsigned nCount = 0;
	for (unsigned i = 0; i < topology.size(); ++i) {
		for (unsigned j = 0; j < topology.at(i); ++j) {
			if (i + 1 >= topology.size()) break;
			for (unsigned k = 0; k < topology.at(i + 1); ++k) {
				m_layers.at(i).at(j).getOutputWeights().at(k).weight = tWeights.at(nCount);
				m_layers.at(i).at(j).getOutputWeights().at(k).deltaWeight = tDeltaWeights.at(nCount);
				cout << "count: " << nCount << " weight: " << m_layers.at(i).at(j).getOutputWeights().at(k).weight
					<< " delta: " << m_layers.at(i).at(j).getOutputWeights().at(k).deltaWeight << endl;
				++nCount;
			}
		}
	}
	inFile.close();
}