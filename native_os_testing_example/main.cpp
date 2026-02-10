#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0]))) // calculates the number of layers (in this case 3)
#define _2_OPTIMIZE 0B00100100 // MULTIPLE_BIASES_PER_LAYER + int8_t quantization
#define _1_OPTIMIZE 0B10011010 // ... + disabled MSE https://github.com/GiorgosXou/NeuralNetworks#define-macro-properties
#define IN_EEPROM_ADDRESS 0    // The position at which the NN will be saved at the internal EEPROM
#define Q_FLOAT_RANGE 2.0      // Our Float32 weights are in the range of [-1,1]
#define USE_RNN__NB            // Makes (vanilla)-RNN the core-architecture of your NeuralNetwork. (NB = NO_BACKPROP support)
#define SELU                   // Defines core activation-function of your NeuralNetwork.

#include <stdio.h>
#include <cstring>
#include "Weights.h"
#include "NeuralNetwork.h"

// wget https://python-course.eu/data/mnist/mnist_train.csv
// wget https://python-course.eu/data/mnist/mnist_test.csv

byte trainImages[60000][28][28];
byte trainLabels[60000];
byte  testImages[10000][28][28];
byte  testLabels[10000];

float *output; // 4th layer's output(s)
const unsigned int layers[] = {28, 5, 8, 10};
NeuralNetwork NN(layers, weights, biases, NumberOf(layers));


void loadDataset(const char* filename, byte (*images)[28][28], byte *labels, unsigned int number){

  FILE * fp = fopen(filename, "r");
  byte * p = &(*images)[0][0];
  char line[4096]; // max should be (4 * (28 * 28) + 4)
  char * pixel;
  
  int i = 0;
  if (fp == NULL)
      exit(EXIT_FAILURE);

  // Read each line
  while (fgets(line, sizeof(line), fp)) {
    i++;
    // Read Label
    *labels++ = (line[0] - 48);

    // Split line per comma
    pixel = strtok(line, ",");
    while ((pixel = strtok(NULL, ",")) != NULL)
        *p++ = atoi(pixel);
  }

  printf("Loaded: %d / %d\n", i, number);
}


unsigned int predictedNumber(){ // is the max output
  unsigned int maxi = 0;
  for (unsigned int i = 0; i < 10; i++)
    if (output[i] > output[maxi])
      maxi = i;
  return maxi;
}


void printResult(const char *type, byte (*images)[28][28], byte *labels, unsigned int number){

  unsigned int count = 0;

  // Go through each image
  for (unsigned int i = 0; i < number; i++){

    // Go through each row of pixels
    for (unsigned int row = 0; row < 28; row++)

      // Feedfowards individually each pixel of the row, and then: returns the output(s).
      for (unsigned int pixel=0; pixel < 28;  pixel++)
        output = NN.FeedForward_Individual(images[i][row][pixel] / 255.0f); // (255 is used to normalize pixels between [0,1])

    // Reset the Internal HiddenStates of the RNN, count and then repeat until all images
    NN.resetStates();
    if (labels[i] == predictedNumber()) count++;
  }

  // Print the overall results
  printf("%s on %d, succesfully predicted %d.\n", type, number, count);
}


void loadDatasets(){
  loadDataset("mnist_train.csv", trainImages, trainLabels, NumberOf(trainLabels));
  loadDataset("mnist_test.csv" ,  testImages,  testLabels, NumberOf( testLabels));
}


void printResults(){
  printResult("Trained", trainImages, trainLabels, NumberOf(trainLabels));
  printResult("Tested ",  testImages,  testLabels, NumberOf( testLabels));
}


int main() {
  loadDatasets();
  printResults();
}

