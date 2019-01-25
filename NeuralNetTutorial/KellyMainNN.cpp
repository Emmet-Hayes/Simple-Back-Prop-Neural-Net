#include "Globalfuncs.h"
#include "LearnData.h"
#include "Net.h"

/* nicely displays values stored in a vector data structure to standard output */
void showVectorVals(string label, vector<double> &v) {
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) cout << v[i] << " ";
	cout << endl;
}

/* reformats a string to be passed into the neural network as vectors of values. 
   NOTE: this function reads specific characters in the input strings to fit the 
   data that is present in the training data. therefore this function must be must
   be changed to fit the data files if they are not labeled like the learndata...*/
void handleStringInput(const string &s, const string &s2, vector<double> &inputVals, vector<double> &targetVals) {
	inputVals.clear();   //clear all garbage left in our test data containers
	targetVals.clear();  //we will refill it in this function
	string label, label2;
	stringstream ss(s), ss2(s2);
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}
	ss2 >> label2;
	if (label2.compare("out:") == 0) {
		double oneValue;
		while (ss2 >> oneValue) {
			targetVals.push_back(oneValue);
		}
	}
}

/* loops until quit is entered, takes user input and interprets it as a choice of options. 
   the Net object is manipulated if the user enters TRAIN, USE and READ. the read and write functions export weights and
   deltaweights to a file to set the network. BE CAREFUL not to try and read a file trained with a different topology or
   the program will call abort() from an assertion failure. */
void mainMenu(LearnData & trainData, vector<unsigned> & topology, Net & myNet, vector<double> & inputVals, vector<double> & targetVals, vector<double> & resultVals) {
	while (true) {                                            //loops until 'quit' is entered
		string comChoice;
		cout << "\nWhat can I do for you today? Commands I understand are as follows:\n(train, use, read, write, quit)\n";
		getline(cin, comChoice);
		trim(comChoice);
		cap(comChoice);
		if (comChoice == "TRAIN") {
			clock_t begin = clock();                          //keeps track of time from the beginning of the program
			unsigned trainingPass = 0;                        //counter for the number of passes through the data that the program traverses
			while (!trainData.isEof()) {                      //while the training data hasn't reached the end of file, keep looping
				++trainingPass;                               //increment counter by 1 each pass
				if (trainData.getNextInputs(inputVals) != topology.at(0)) 
					break;                                    //Get next input data, if its not equal to topology of input layer, leave the loop
				myNet.feedForward(inputVals);                 //feed values from the inputVals vector forward through the network
				myNet.getResults(resultVals);                 //Collect the net's actual output result.
				trainData.getTargetOutputs(targetVals);       //get the target outputs from the file and save it in the vector
				assert(targetVals.size() == topology.back()); //assert that the size of the vector of target values matches the last value in the topology
				myNet.backProp(targetVals);                   //use the target values to perform back propagation (calculating gradients)
				if (trainingPass % 500 == 0) {                //print out results and metrics every 500 passes
					cout << endl << "Pass " << trainingPass;
					showVectorVals(": Inputs:", inputVals);
					showVectorVals("Outputs:", resultVals);
					showVectorVals("Targets:", targetVals);
					cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
				}
			}
			clock_t end = clock();                            //time stamp the end of the loop
			cout << "Total time spent learning: " << double(end - begin) / CLOCKS_PER_SEC << " secs.\n";
			cout << "\n\n\nFINAL SCORES FROM THE NEURAL NETWORK\n\n";
			cout << "Passes: " << trainingPass << endl;
			showVectorVals("Inputs: ", inputVals);
			showVectorVals("Target Outputs: ", targetVals);
			showVectorVals("Network Outputs: ", resultVals);
			cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
		}
		else if (comChoice == "USE") {
			string userIn, userOut;
			cout << "Add the input and output data you'd like to test, please.\n(YOU MUST TYPE in: and then single digit values IN ORDER TO WORK):\n"
				"in: 1 0 1\nout: 1 0\n";
			getline(cin, userIn);
			getline(cin, userOut);
			try {
				handleStringInput(userIn, userOut, inputVals, targetVals);
				myNet.feedForward(inputVals);
				myNet.getResults(resultVals);
				showVectorVals("Inputs: ", inputVals);
				showVectorVals("Network Outputs: ", resultVals);
				cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
			}
			catch (invalid_argument i) { cout << "The arguments you provided could not be converted to integer values\n"; }
			catch (out_of_range o) { cout << "The arguments you provided could not be converted because they were out of range.\n"; }
			catch (exception e) { cout << "The arguments you provided caused an error and could not be converted.\n"; }
		}
		else if (comChoice == "READ") {
			myNet.readNet(topology);
		}
		else if (comChoice == "WRITE") {
			myNet.writeNet(topology);
		}
		else if (comChoice == "QUIT") { break; }
		else {
			cout << "Couldn't understand your command. Please enter train, use, read, write, or quit.\n";
			break;
		}
	}
}

/* main holds the training data, and makes a single call to mainMenu which encapsulates each function the network is 
   capable of executing. */
int main() {
	LearnData trainData("learnData.txt");                                    //make a stack object for training data, URL for the file name
	vector<unsigned> topology;                                               //make a vector of unsigned integers for storing the topology of the network ( ie. topology: 8 6 3 )
	trainData.getTopology(topology);                                         //get the topology information from the training data file and store it in the topology vector
	Net myNet(topology);                                                     //create a neural network with the topology from the file
	vector<double> inputVals, targetVals, resultVals;                        //declaring vectors of real numbers to store inputs, target outputs, and resulting outputs from the network
	cout << "HELLO MY NAME IS beaver AND I AM ALIIIIIIVEEE HAHAHAHAHA DESTROY ALL HUMANS\n"
		"Lol I'm just a Neural Network Machine Learning algorithm based on supervised learning." 
		"\nI'll do my best to try and learn something from the input and output data you supplied in the learnData.txt file.";
	mainMenu(trainData, topology, myNet, inputVals, targetVals, resultVals); //main menu function encapulsates the rest of the program
	cout << "Press any key to end the program...\n";
	cin.ignore();
	return 0;
}