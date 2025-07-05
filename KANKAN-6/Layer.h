#pragma once
#include <memory>
#include <vector>
#include <algorithm>
#include "Urysohn.h"

class Layer {
public:
	Layer(int nUrysohns, std::vector<double> xmin, std::vector<double> xmax, double targetMin, double targetMax, int nPoints) {
		if (xmin.size() != xmax.size()) {
			printf("Fatal: sizes of xmin, xmax or nFunctions mismatch\n");
			exit(0);
		}
		for (int i = 0; i < nUrysohns; ++i) {
			_urysohns.push_back(std::make_unique<Urysohn>(xmin, xmax, targetMin, targetMax, nPoints));
		}
	}
	Layer(int nUrysohns, int nFunctions, double targetMin, double targetMax, int nPoints) {
		for (int i = 0; i < nUrysohns; ++i) {
			_urysohns.push_back(std::make_unique<Urysohn>(targetMin, targetMax, nFunctions, nPoints));
		}
	}
	Layer(const Layer& layer) {
		_urysohns.clear();
		_urysohns = std::vector<std::unique_ptr<Urysohn>>(layer._urysohns.size());
		for (int i = 0; i < layer._urysohns.size(); ++i) {
			_urysohns[i] = std::make_unique<Urysohn>(*layer._urysohns[i]);
		}
	}
	void Input2Output(const std::vector<double>& input, std::vector<double>& output, bool freezeModel = true) {
		for (int i = 0; i < _urysohns.size(); ++i) {
			output[i] = _urysohns[i]->GetUrysohn(input, freezeModel);
		}
	}
	void Input2Output(const std::vector<double>& input, std::vector<double>& output,
		std::vector<std::vector<double>>& derivatives, bool freezeModel = true) {
		for (int i = 0; i < _urysohns.size(); ++i) {
			output[i] = _urysohns[i]->GetUrysohn(input, derivatives[i], freezeModel);
		}
	}
	void ComputeDeltas(const std::vector<std::vector<double>>& derivatives, const std::vector<double>& deltasIn,
		std::vector<double>& deltasOut) {
		std::fill(deltasOut.begin(), deltasOut.end(), 0.0);
		int nRows = (int)derivatives[0].size();
		int nCols = (int)derivatives.size();
		for (int n = 0; n < nRows; ++n) {
			for (int k = 0; k < nCols; ++k) {
				deltasOut[n] += derivatives[k][n] * deltasIn[k];
			}
		}
	}
	void Update(const std::vector<double>& deltas) {
		for (int i = 0; i < _urysohns.size(); ++i) {
			_urysohns[i]->Update(deltas[i]);
		}
	}
	void IncrementPoins() {
		for (int i = 0; i < _urysohns.size(); ++i) {
			_urysohns[i]->IncrementPoints();
		}
	}
private:
	std::vector<std::unique_ptr<Urysohn>> _urysohns;
};

