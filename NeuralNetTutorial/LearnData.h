#pragma once
#ifndef LearnData_H
#define LearnData_H
#include "Globalfuncs.h"
using namespace std; //lets us use everything in the std namespace without the scoping operator (ie std::vector<T>)
class LearnData {    //holds the data file that the network will train on
public:
	LearnData(const string fileName);
	bool isEof() { return m_trainingDataFile.eof(); }
	void getTopology(vector<unsigned> &topology);
	unsigned getNextInputs(vector<double> &inputVals); // Returns the number of input values read from the file:
	unsigned getTargetOutputs(vector<double> &targetOutputVals);
private:
	ifstream m_trainingDataFile;
};
#endif