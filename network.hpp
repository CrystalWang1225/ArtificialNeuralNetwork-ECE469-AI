//
//  network.hpp
//  network
//
//  Created by Crystal Wang on 12/7/19.
//  Copyright © 2019 Crystal Wang. All rights reserved.
//

#ifndef network_hpp
#define network_hpp

#include <stdio.h>
#include <vector>
#include <fstream>

using namespace std;

class network{
public:
    network(ifstream &initial);
    int training(ifstream &training, double learnRate, int epochs);
    int test(ifstream &test, ofstream &output);
    void save(ostream &output);
private:
    class neuron;
    class link{
    public:
        double weight;
        neuron *connected;
    };
    class neuron{
    public:
        double inputValue;
        double activation;
        double error;
        vector<link> inLink;
        vector<link> outLink;
    };
    
    class trainingSet{
    public:
        vector<double> input;
        vector<int> output;
    };
     double activationFunction(double inputValue);
    double activationDerivative(double inputValue);
    int numLayer;
    vector<int> layerSize;
    vector<vector<neuron> > layer;
};

#endif /* network_hpp */
