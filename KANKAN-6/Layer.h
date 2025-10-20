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
		size_t nFunctions = (int)xmin.size();
		_derivatives = std::vector<std::vector<double>>(nUrysohns);
		for (int i = 0; i < nUrysohns; ++i) {
			_derivatives[i] = std::vector<double>(nFunctions);
		}
	}
	Layer(int nUrysohns, int nFunctions, double targetMin, double targetMax, int nPoints) {
		for (int i = 0; i < nUrysohns; ++i) {
			_urysohns.push_back(std::make_unique<Urysohn>(targetMin, targetMax, nFunctions, nPoints));
		}
		_derivatives = std::vector<std::vector<double>>(nUrysohns);
		for (int i = 0; i < nUrysohns; ++i) {
			_derivatives[i] = std::vector<double>(nFunctions);
		}
	}
	Layer(const Layer& layer) {
		_urysohns.clear();
		_urysohns = std::vector<std::unique_ptr<Urysohn>>(layer._urysohns.size());
		for (size_t i = 0; i < layer._urysohns.size(); ++i) {
			_urysohns[i] = std::make_unique<Urysohn>(*layer._urysohns[i]);
		}
		_derivatives.clear();
		_derivatives = std::vector<std::vector<double>>(layer._derivatives.size());
		for (size_t i = 0; i < layer._derivatives.size(); ++i) {
			_derivatives[i] = std::vector<double>(layer._derivatives[i].size());
		}
	}
	void Input2Output(const std::vector<double>& input, std::vector<double>& output, bool freezeModel = true, bool computeDerivative = false) {
		if (computeDerivative) {
			for (size_t i = 0; i < _urysohns.size(); ++i) {
				output[i] = _urysohns[i]->GetUrysohn(input, _derivatives[i], freezeModel);
			}
		}
		else {
			for (size_t i = 0; i < _urysohns.size(); ++i) {
				output[i] = _urysohns[i]->GetUrysohn(input, freezeModel);
			}
		}
	}
	void ComputeDeltas(const std::vector<double>& deltasIn, std::vector<double>& deltasOut) {
		std::fill(deltasOut.begin(), deltasOut.end(), 0.0);
		size_t nRows = (int)_derivatives[0].size();
		size_t nCols = (int)_derivatives.size();
		for (size_t n = 0; n < nRows; ++n) {
			for (size_t k = 0; k < nCols; ++k) {
				deltasOut[n] += _derivatives[k][n] * deltasIn[k];
			}
		}
	}
	void Update(const std::vector<double>& deltas) {
		for (size_t i = 0; i < _urysohns.size(); ++i) {
			_urysohns[i]->Update(deltas[i]);
		}
	}
	void IncrementPoins() {
		for (size_t i = 0; i < _urysohns.size(); ++i) {
			_urysohns[i]->IncrementPoints();
		}
	}
	void ShowData() {
		for (size_t i = 0; i < _urysohns.size(); ++i) {
			_urysohns[i]->ShowData();
		}
	}
	std::vector<double> GetAllMinValues(size_t n) {
		if (n >= _urysohns.size()) {
			printf("Fatal: size min mismatch");
			exit(0);
		}
		return _urysohns[n]->GetAllMinValues();
	}
	std::vector<double> GetAllMaxValues(size_t n) {
		if (n >= _urysohns.size()) {
			printf("Fatal: size max mismatch");
			exit(0);
		}
		return _urysohns[n]->GetAllMaxValues();
	}
	void SetMinMaxAllU(double min, double max, size_t n) {
		if (n >= _urysohns.size()) {
			printf("Fatal: size _u mismatch");
			exit(0);
		}
		_urysohns[n]->SetAllMinMax(min, max);
	}
	void RenormalizeAllU(const std::vector<double>& min, const std::vector<double>& max, double wantedMin, double wantedMax) {
		if (min.size() != max.size() || min.size() != _urysohns.size()) {
			printf("Fatal: sizes mismatch");
			exit(0);
		}
		for (size_t i = 0; i < _urysohns.size(); ++i) {
			double a = (max[i] - min[i]) / (wantedMax - wantedMin);
			double b = min[i] - a * wantedMin;
			_urysohns[i]->Renormalize(b, a);
		}
	}
private:
	std::vector<std::unique_ptr<Urysohn>> _urysohns;
	std::vector<std::vector<double>> _derivatives; 
};

