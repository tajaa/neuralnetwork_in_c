#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// simple NN that can learn xor func

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dsigmoid(double x) { return x * (1 - x); }

double init_weights() { return ((double)rand()) / ((double)RAND_MAX); }

// for shuffling data
void shuffle(int *array, size_t n) {
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

#define numInputs 2
#define numHiddenNodes 2
#define numOutPuts 1
#define numTrainingSets 4

int main(void) {
  // lr = learning rate
  const double lr = 0.1f;

  double hiddenLayer[numHiddenNodes];
  double outputLayer[numOutPuts];

  double hiddenLayerBias[numHiddenNodes];
  double outputLayerBias[numOutPuts];

  double hiddenWeights[numInputs][numHiddenNodes];
  double outputWeights[numHiddenNodes][numOutPuts];

  double training_inputs[numTrainingSets][numInputs] = {
      {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
  double training_outputs[numTrainingSets][numOutPuts] = {
      {0.0f}, {1.0f}, {1.0f}, {0.0f}};

  // init weights
  for (int i = 0; i < numInputs; i++) {
    for (int j = 0; j < numHiddenNodes; j++) {
      hiddenWeights[i][j] = init_weights();
    }
  }

  for (int i = 0; i < numHiddenNodes; i++) {
    for (int j = 0; j < numOutPuts; j++) {
      outputWeights[i][j] = init_weights();
    }
  }

  for (int i = 0; i < numOutPuts; i++) {
    outputLayerBias[i] = init_weights();
  }

  int trainingSetOrder[] = {0, 1, 2, 3};

  int numberOfEpochs = 10000;

  // train neural network for a num of epochs
  for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
    shuffle(trainingSetOrder, numTrainingSets);

    for (int x = 0; x < numTrainingSets; x++) {
      int i = trainingSetOrder[x];

      // forward pass

      // compute hidden layer activation
      for (int j = 0; j < numHiddenNodes; j++) {
        double activation = hiddenLayerBias[j];

        for (int k = 0; k < numInputs; k++) {
          activation += training_inputs[i][k] * hiddenWeights[k][j];
        }
        hiddenLayer[j] = sigmoid(activation);
      }
      // compute output layer activation
      for (int j = 0; j < numOutPuts; j++) {
        double activation = outputLayerBias[j];
        for (int k = 0; k < numHiddenNodes; k++) {
          activation += hiddenLayer[k] * outputWeights[k][j];
        }
        outputLayer[j] = sigmoid(activation);
      }
      // print the reults from forward pass
      printf("input: %g %g Output: %g Predicted Output: %g \n",
             training_inputs[i][0], training_inputs[i][1], outputLayer[0],
             training_outputs[i][0]);

      // backprop

      // compute change in output weights
      double deltaOutput[numOutPuts];

      for (int j = 0; j < numOutPuts; j++) {
        double error = (training_outputs[i][j] - outputLayer[j]);
        deltaOutput[j] = error * dsigmoid(outputLayer[j]);
      }
      // compute the change in hidden weights
      double deltaHidden[numHiddenNodes];
      for (int j = 0; j < numHiddenNodes; j++) {
        double error = 0.0f;
        for (int k = 0; k < numOutPuts; k++) {
          // change in output weights
          error += deltaOutput[k] * outputWeights[j][k];
        }
        deltaHidden[j] = error * dsigmoid(hiddenLayer[j]);
      }
      // Apply changes in output weights
      for (int j = 0; j < numOutPuts; j++) {
        outputLayerBias[j] += deltaOutput[j] * lr;

        for (int k = 0; k < numHiddenNodes; k++) {
          outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
        }
      }
      // apply change in hidden weights
      for (int j = 0; j < numHiddenNodes; j++) {
        hiddenLayerBias[j] += deltaHidden[j] * lr;
        for (int k = 0; k < numInputs; k++) {
          hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
        }
      }

      // print final weights after done training
      fputs("final hidden weights\n[", stdout);
      for (int j = 0; j < numHiddenNodes; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numInputs; k++) {
          printf("%f", hiddenWeights[k][j]);
        }
        fputs("] ", stdout);
      }

      fputs("]\n final hiden biases\n[", stdout);
      for (int j = 0; j < numHiddenNodes; j++) {
        printf("%f ", hiddenLayerBias[j]);
      }

      fputs("final output weights\n[", stdout);
      for (int j = 0; j < numOutPuts; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numHiddenNodes; k++) {
          printf("%f", outputWeights[k][j]);
        }
        fputs("] \n", stdout);
      }

      fputs("]\n final output biases\n[", stdout);
      for (int j = 0; j < numOutPuts; j++) {
        printf("%f ", outputLayerBias[j]);
      }
      fputs("] \n", stdout);
    }
  }
  return 0;
}
