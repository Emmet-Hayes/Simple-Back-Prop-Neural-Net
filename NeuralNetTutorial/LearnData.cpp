#include "LearnData.h"

/* makes a LearnData object with passed file name. */
LearnData::LearnData(const string filename) {
	m_trainingDataFile.open(filename.c_str());
}

/* reads the top line of the file and saves topology values in vector. */
void LearnData::getTopology(vector<unsigned> &topology) {
	string line, label;
	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0) abort();
	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
}

/* each of these learndata member functions are variations of the same method.
   clears and fills the values in the vector passed. */
unsigned LearnData::getNextInputs(vector<double> &inputVals) {
	inputVals.clear();
	string line, label;
	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}
	return inputVals.size();
}

/* just like above function but for target output values. */
unsigned LearnData::getTargetOutputs(vector<double> &targetOutputVals) {
	targetOutputVals.clear();
	string line, label;
	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}
	return targetOutputVals.size();
}