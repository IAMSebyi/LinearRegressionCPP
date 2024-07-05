#pragma once

#include <iostream>
#include <vector>

class LinearRegression
{
public:
	// CONSTRUCTOR
	LinearRegression(const std::vector<std::vector<float>>& features, const std::vector<float>& targets, const int& numOfFeatures, const int& numOfDataPoints); 
	
	// METHODS
	float Predict(std::vector<float> input) const;
	void Train(float learningRate, int maxIterations = 7000);
	std::vector<float> GetParameters() const;

private:
	// METHODS
	float Cost() const; // squared error cost function
	std::vector<float> CoeffGradient() const; // gradient of the cost function with respect to coeffiecents parameter
	float InterceptGradient() const; // gradient of the cost function with respect to intercept parameter

	// DATA POINTS
	std::vector<std::vector<float>> features; // input independent variables
	std::vector<float> targets; // output dependent variables

	// MODEL PARAMETERS
	std::vector<float> coefficients;
	float intercept;

	// VARIABLES
	int numOfFeatures;
	int numOfDataPoints;
};

