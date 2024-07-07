#pragma once

#include <iostream>
#include <vector>
#include <cmath>

class LinearRegression
{
public:
	// Constructor
	LinearRegression(const std::vector<std::vector<float>>& features, const std::vector<float>& targets, const int& numOfFeatures, const int& numOfDataPoints); 
	
	// Methods
	float Predict(std::vector<float> input, const bool test = false) const;
	void Train(const float& learningRate, const int maxIterations = 40000);
	std::vector<float> GetParameters() const;

private:
	// Methods
	void Scale();
	float Cost() const;
	std::vector<float> CoeffGradient() const;
	float InterceptGradient() const;
	std::vector<float> GetMean() const;
	std::vector<float> GetStandardDeviation(const std::vector<float>& mean) const;

	// Data members
	std::vector<std::vector<float>> features;
	std::vector<float> targets;

	// Model parameters
	std::vector<float> coefficients;
	float intercept;

	// Variables
	int numOfFeatures;
	int numOfDataPoints;
	std::vector<float> mean;
	std::vector<float> standardDeviation;
};

