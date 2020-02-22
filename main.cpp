#include <iostream>
#include <fstream>
#include "network.hpp"
#include <string>
using namespace std;

int main(int argc, const char * argv[]) {
    double learningRate;
    int epoch;
    string trainingSet, output, initial, trained,testSet;
    string choice;
    ifstream initialFile,trainingSetFile, trainedFile, testSetFile;
    ofstream outputFile;
    cout << "Please choose to test the neural network or train your samples(test/train): "<< endl;
    cin >> choice;
    if (choice == "test" || choice == "Test"){

    cout << " Please enter the trained neural network file : ";
    cin >> trained;
    cout << "Please enter the traning set file : ";
    cin >> testSet;
    cout << "Please enter the output file : ";
    cin >> output;
    trainedFile.open(trained.c_str());
    testSetFile.open(testSet.c_str());
    outputFile.open(output.c_str());
    if (trainedFile.is_open() && testSetFile.is_open() && outputFile.is_open()){
        network *test = new network(trainedFile);
        test -> test(testSetFile, outputFile);
    }
    else {
        cout << "Unable to open one of the files above" << endl;
    }
    }
    else if (choice == "train" || choice == "Train"){
        cout << "Enter the initial weight file: ";
        cin >> initial;
        cout << "Enter the training examples : ";
        cin >> trainingSet;
        cout << "Enter the output file : ";
        cin >> output;
        cout << "Enter the learning rate: ";
        cin >> learningRate;
        cout << "Enter the number of epochs: ";
        cin >> epoch;
        initialFile.open(initial.c_str());
        trainingSetFile.open(trainingSet.c_str());
        outputFile.open(output.c_str());
        if (initialFile.is_open() && trainingSetFile.is_open() && outputFile.is_open()){
            network *test = new network(initialFile);
            test -> training(trainingSetFile, learningRate, epoch);
            test -> save(outputFile);
        }
        else {
            cout << "Unable to open one of the files above" << endl;
        }
    }
    else {
        cout << "Wrong input, please restart the program" << endl;
    }
    return 0;
}
