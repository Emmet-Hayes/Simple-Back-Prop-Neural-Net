#pragma once
#include <string>    //to manipulate the C++ string class
#include <vector>    //allows us to use a modern array with a variable length and member functions (vector<T>)
#include <iostream>  //input and output to the console (cin >>, cout <<)
#include <iomanip>   //formatting input and output  (setprecision, fixed)
#include <cstdlib>   //for the c standard library, used for random number generation (rand() and srand(unsigned))
#include <cassert>   //for the assert function, useful for testing to make sure objects arent broken
#include <cmath>     //for the tanh() function, which we use in the transfer of neurons to have new weights
#include <fstream>   //for making input file and output file objects and reading and writing data to them.
#include <sstream>   //for appending numeric type variables to a string stream that can easily convert to string
#include <ctime>     //for tracking system time (ie time spent on learning from the learning data file).
#include <algorithm> //gives us access to super efficient algorithms to deal with strings and containers (like a vector)
using namespace std; //we can use anything from the std namespace without having to scope (std::)

/* trims the left side of any string */
static inline void ltrim(string &s) {
	s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

/* trims the right side of any string */
static inline void rtrim(string &s) {
	s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

/* trims both sides of any string (calls ltrim and rtrim) */
static inline void trim(string &s) {
	ltrim(s);
	rtrim(s);
}

/* capitalizes each character in the string, all non-alpha characters unaffected. */
static inline void cap(string &s) {
	for (unsigned int i = 0; i < s.size(); ++i) {
		s.at(i) = toupper(s.at(i));
	}
}