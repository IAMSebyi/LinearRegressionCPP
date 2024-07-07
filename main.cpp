#include "LinearRegression.h"
#include <fstream>

int main() {
	// Open the input and output files
	std::ifstream inputData("data.txt");
	std::ifstream inputTest("test.txt");
	std::ofstream outputParameters("parameters.txt");
	
	int numOfFeatures, numOfDataPoints, maxIterations = -1;
	float learningRate;

	// Read number of features and data points
	inputData >> numOfFeatures >> numOfDataPoints;

	// Initialize vectors for features and targetsx
	std::vector<std::vector<float>> features (numOfDataPoints, std::vector<float> (numOfFeatures)); // Input independent variables
	std::vector<float> targets (numOfDataPoints); // Output dependent variables

	// Read features and targets from input file
	for (int i = 0; i < numOfDataPoints; i++) {
		for (int j = 0; j < numOfFeatures; j++) {
			inputData >> features[i][j];
		}
		inputData >> targets[i];
	}

	// Read learning rate and max iterations
	inputData >> learningRate >> maxIterations;

	// Create LinearRegression model
	LinearRegression model(features, targets, numOfFeatures, numOfDataPoints);

	// Train the model
	if (maxIterations == -1) model.Train(learningRate);
	else model.Train(learningRate, maxIterations);

	// Get the model parameters (coefficients and intercept)
	std::vector<float> parameters;
	parameters = model.GetParameters();

	// Write the model parameters to output file
	for (const float& param : parameters) {
		outputParameters << param << " ";
	}

	// Check if test data points are provided
	int numOfTestDataPoints = -1;
	inputTest >> numOfTestDataPoints;

	// Predict and print results for test data points
	if (numOfTestDataPoints != -1) {
		std::vector<float> testFeature (numOfFeatures);

		for (int i = 0; i < numOfTestDataPoints; i++) {
			std::cout << "Result for input { ";

			for (int j = 0; j < numOfFeatures; j++) {
				inputTest >> testFeature[j];
				std::cout << testFeature[j] << " ";
			}

			std::cout << "}: " << model.Predict(testFeature, true) << '\n';
		}
	}

	return 0;
}