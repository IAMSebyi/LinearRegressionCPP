#include "LinearRegression.h"

// Constructor to initialize the LinearRegression object with features, targets, and dimensions
LinearRegression::LinearRegression (const std::vector<std::vector<float>>& features, const std::vector<float>& targets, const int& numOfFeatures, const int& numOfDataPoints)
	: features(features), targets(targets), numOfFeatures(numOfFeatures), numOfDataPoints(numOfDataPoints), coefficients(features[0].size(), 0), intercept(0)
{
}

// Predict the target value for a given input vector
float LinearRegression::Predict(std::vector<float> input) const
{
	float result = intercept;

	for (int i = 0; i < numOfFeatures; i++) {
		result += input[i] * coefficients[i];
	}

	return result;
}

// Calculate the squared error cost function
float LinearRegression::Cost() const
{
	float result = 0;

	for (int i = 0; i < numOfDataPoints; i++) {
		float error = Predict(features[i]) - targets[i];
		result += error * error;
	}

	result /= 2 * numOfDataPoints;

	return result;
}

// Calculate the gradient of the cost function with respect to coefficients
std::vector<float> LinearRegression::CoeffGradient() const
{
	std::vector<float> result (numOfFeatures);

	for (int i = 0; i < numOfFeatures; i++) {
		float sum = 0;
		for (int j = 0; j < numOfDataPoints; j++) {
			sum += (Predict(features[j]) - targets[j]) * features[j][i];
		}
		result[i] = sum / numOfDataPoints;
	}

	return result;
}

// Calculate the gradient of the cost function with respect to intercept
float LinearRegression::InterceptGradient() const
{
	float result = 0;

	for (int i = 0; i < numOfDataPoints; i++) {
		result += Predict(features[i]) - targets[i];
	}
	result /= numOfDataPoints;

	return result;
}

// Train the model using Gradient Descent
void LinearRegression::Train(float learningRate, int maxIterations)
{
	float prevCost;
	std::vector<float> prevCoefficients;
	float prevIntercept;

	for (int i = 0; i < maxIterations; i++) {
		prevCost = Cost();
		prevCoefficients = coefficients;
		prevIntercept = intercept;

		std::vector<float> coeffSlope = CoeffGradient();
		float interceptSlope = InterceptGradient();

		for (int j = 0; j < numOfFeatures; j++) {
			coefficients[j] -= learningRate * coeffSlope[j];
		}

		intercept -= learningRate * interceptSlope;

		float currentCost = Cost();
		if (i > 0 && currentCost > prevCost) {
			// EARLY STOPPAGE DUE TO INCREASE IN LOSS (CONVERGENCE PROBABLY REACHED)
			coefficients = prevCoefficients;
			intercept = prevIntercept;
			return;
		}
	}
}

// Get the model parameters (coefficients and intercept)
std::vector<float> LinearRegression::GetParameters() const
{
	std::vector<float> parameters (coefficients);
	parameters.push_back(intercept);

	return parameters;
}
