
#include "network.hpp"
#include <iostream>
#include <cmath>


using namespace std;

double A ,B,C,D;
double overall, precision, recall, f1;
double avOverall, avPrecision, avRecall, avF1;

network::network(ifstream &initial){
    this -> numLayer = 3; //setting the networks to have exact three layers
    this -> layerSize.resize(this -> numLayer);
    this -> layer.resize(this -> numLayer);
    
    for (int i = 0;i < numLayer; i++){
        initial >> this -> layerSize[i];
        this -> layerSize[i] ++;
        this -> layer[i].resize(this -> layerSize[i]);
    }
    //for w0 it is set to -1
    for (int i = 0; i < this -> numLayer; i++){
        this -> layer[i][0].activation = -1;
    }
    
    for (int current = 0; current< this -> numLayer - 1; current ++){
        for (int i = 1; i < this -> layerSize[current + 1]; i++){
            for (int j = 0; j < this -> layerSize[current]; j++){
                double weight;
                initial >> weight;
                link inLink, outLink;
                outLink.weight = weight;
                outLink.connected = &this -> layer[current + 1][i];
                this -> layer[current][j].outLink.push_back(outLink);
                
                inLink.weight = weight;
                inLink.connected = &this -> layer[current][j];
                this -> layer[current+1][i].inLink.push_back(inLink);
            }
        }
    }
}

void network::save(ostream &output){
    output << setprecision(3) << fixed;
    for (int l = 0; l < this -> numLayer; l++){
        if (l != 0){
            output << " ";
        }
        output << this -> layerSize[l]-1;
    }
    output << endl;
    for (int i = 1; i < this -> numLayer; i ++){
        for (int j = 1; j < this -> layerSize[i]; j++){
            vector<link>::iterator it;
            for (it = this -> layer[i][j].inLink.begin(); it != this -> layer[i][j].inLink.end(); it ++){
                if (it != this -> layer[i][j].inLink.begin()){
                    output << " ";
                }
                output << it -> weight;
            }
            output << endl;
        }
    }
}


//the Sigmoid function
double network::activationFunction(double inputValue){
    return 1.000 / (1.000 + exp(-inputValue));
}

double network::activationDerivative(double inputValue){
    return this -> activationFunction(inputValue) * (1.000 - this -> activationFunction(inputValue));
}

//The training part contains Back Propagation
int network::training(ifstream &training, double learnRate,  int epoch){
    vector<trainingSet> trainingSet;
    int inputN, outputN, setN;
  
    training >> setN >> inputN >> outputN;
    trainingSet.resize(setN);
    for (int i = 0; i < setN; i++){
        trainingSet[i].input.resize(inputN);
        trainingSet[i].output.resize(outputN);
        for(int j = 0; j < inputN; j++){
            training >> trainingSet[i].input[j];
        }
      for(int k = 0; k < outputN; k++){
            training >> trainingSet[i].output[k];
      }
}
    
      int outputI = this -> numLayer -1;
    for (int i = 0; i < epoch; i++){
        for (int c = 0; c < setN; c++){
            for (int j = 0; j < inputN; j++){
                //copying training samples to input nodes of the network
                this -> layer[0][j+1].activation = trainingSet[c].input[j];
            }
            for (int l = 1; l < this -> numLayer; l++){
                for (int n = 1; n < this -> layerSize[l]; n++){
                    this -> layer[l][n].inputValue = 0;
                    vector<link>::iterator it;
                    for (it = this -> layer[l][n].inLink.begin(); it != this -> layer[l][n].inLink.end(); it++){
                        this -> layer[l][n].inputValue += it->weight * it->connected->activation;
                    }
                    this ->layer[l][n].activation = this -> activationFunction(this -> layer[l][n].inputValue);
                }
            }
            //errors from output network to the input layer
            for (int n = 1; n < this ->layerSize[outputI]; n++ ){
                this -> layer[outputI][n].error = this -> activationDerivative(this -> layer[outputI][n]. inputValue) * (trainingSet[c].output[n -1] -  this -> layer[outputI][n].activation);
            }
            for (int l = outputI -1; l > 0; l--){
                for (int ii = 1; ii < this -> layerSize[l]; ii ++){
                    double sum = 0;
                    vector<link>::iterator it;
                    for (it = this -> layer[l][ii].outLink.begin();it!= this -> layer[l][ii].outLink.end(); it++){
                        sum += it -> weight * it -> connected -> error;
                    }
                    this -> layer[l][ii].error = this -> activationDerivative(this -> layer[l][ii].inputValue) * sum;
                }
            }
            for (int l = 1; l < this -> numLayer; l++){
                for (int j = 1; j < this -> layerSize[l]; j++){
                    vector<link>::iterator it;
                    for (it = this -> layer[l][j].inLink.begin(); it != this -> layer[l][j].inLink.end(); it ++){
                        it -> weight = it -> weight + learnRate * it -> connected -> activation * this ->layer[l][j].error;
                        it -> connected -> outLink[j -1].weight = it -> weight;
                    }
                }
            }
            
        }
    }
        return 0;
}

int network::test(ifstream &test, ofstream &output){
    int setN, inputN, outputN;
    vector<trainingSet> example;
    vector<vector<double> > result;
    test >> setN >> inputN >> outputN;
    example.resize(setN);
    result.resize(outputN);
    
    for(int i = 0; i < setN; i++){
        example[i].input.resize(inputN);
         example[i].output.resize(outputN);
        for (int j = 0; j < inputN; j++){
            test >> example[i].input[j];
        }
        for (int k = 0; k < outputN; k++ ){
            test >> example[i].output[k];
            if (i == 0){
                result[k].resize(4);
                for (int m = 0; m < 4; m++){
                    result[k][m] = 0;
                }
            }
        }
    }
    
    int outputI = this -> numLayer -1;
    for (int c = 0; c < setN; c++){
        for (int i = 0; i < inputN; i++){
            this -> layer[0][i+1].activation = example[c].input[i];
        }
        for (int l = 1; l < this -> numLayer; l++){
            for (int j = 1; j < this -> layerSize[l]; j ++ ){
                this -> layer[l][j].inputValue = 0;
                vector<link>::iterator it;
                for (it = this -> layer[l][j]. inLink.begin(); it != this -> layer[l][j].inLink.end(); it ++){
                    this -> layer[l][j].inputValue += it -> weight * (it -> connected-> activation);
                }
                this -> layer[l][j].activation = this -> activationFunction(this -> layer[l][j].inputValue);
            }
        }
        for (int n = 1; n < this -> layerSize[outputI]; n ++ ){
            if (this -> layer[outputI][n].activation >= 0.5){
                if (example[c].output[n-1]){
                    result[n-1][0]++;
                }
                else {
                    result[n-1][1]++;
                }
            }
            else {
                if (example[c].output[n-1]){
                    result[n-1][2]++;
                }
                else {
                    result[n-1][3]++;
                }
            }
        }
    }
    
   //setting the precision to three decimal places
    output << setprecision(3) << fixed;
    A = 0;B = 0;C = 0;D = 0;
    for (int i = 0; i < outputN; i++){
        A += result[i][0];
        B += result[i][1];
        C += result[i][2];
        D += result[i][3];
        output << (int)result[i][0] << " " << (int)result[i][1] << " "<<(int)result[i][2] << " " <<(int)result[i][3] << " ";
        overall = (result[i][0] + result[i][3])/(result[i][0]+result[i][1]+result[i][2]+result[i][3]);
        precision = result[i][0] / (result[i][0] + result[i][1]);
        recall = result[i][0] / (result[i][0] + result[i][2]);
        f1 = (2*precision*recall)/ (precision + recall);
        if (overall != overall) overall = 0;
        if (precision != precision) precision = 0;
        if (recall != recall)recall = 0;
        if (f1 != f1)f1 = 0;
        output << overall << " " << precision << " "<< recall<< " "<< f1 << endl;
        
        avOverall +=overall;
        avPrecision +=precision;
        avRecall += recall;
        }
    
       //Micro-averaging
    overall = (A + D)/(A + B + C + D);
    precision = A/(A+B);
    recall = A/(A+C);
    f1 = (2 * precision * recall)/(precision + recall);
    if (overall != overall) overall = 0;
    if (precision != precision) precision = 0;
    if (recall != recall)recall = 0;
    if (f1 != f1)f1 = 0;
    output << overall << " " << precision << " " << recall << " " << f1 << endl;
    //Macro-Averaging
    avOverall /=outputN;
    avPrecision /=outputN;
    avRecall /=outputN;
    avF1 = (2*avPrecision*avRecall)/(avPrecision + avRecall);
    if (avF1 != avF1)avF1 = 0;
    output << avOverall<< " " << avPrecision << " " << avRecall << " " << avF1 << endl;
    return 0;
    }

