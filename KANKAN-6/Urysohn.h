#pragma once
#include <memory>
#include <vector>
#include <random>
#include <ctime>

class Urysohn {
public:
	Urysohn(const std::vector<double>& argmin, const std::vector<double>& argmax, double umin, double umax, int nPoints) {
		if (argmin.size() != argmax.size()) {
			printf("Fatal: argument sizes mismatch");
			exit(0);
		}
		int nFunctions = (int)argmin.size();
		_model = std::vector<std::vector<double>>(nFunctions);
		std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
		std::uniform_real_distribution<double> dist(umin, umax);
		for (int i = 0; i < nFunctions; ++i) {
			_model[i] = std::vector<double>(nPoints);
			for (int j = 0; j < nPoints; ++j) {
				_model[i][j] = dist(rng);
			}
		}
		_xmin = std::vector<double>(nFunctions);
		_xmax = std::vector<double>(nFunctions);
		_deltax = std::vector<double>(nFunctions);
		_offset = std::vector<double>(nFunctions);
		_index = std::vector<int>(nFunctions);
		for (int i = 0; i < nFunctions; ++i) {
			_xmin[i] = argmin[i];
			_xmax[i] = argmax[i];
			SetLimits(i);
		}
	}
	Urysohn(double umin, double umax, int nFunctions, int nPoints) {
		srand((unsigned int)time(nullptr));
		_model = std::vector<std::vector<double>>(nFunctions);
		for (int i = 0; i < nFunctions; ++i) {
			_model[i] = std::vector<double>(nPoints);
			for (int j = 0; j < nPoints; ++j) {
				_model[i][j] = (rand() % 1000 / 1000.0) * (umax - umin) + umin;
			}
		}
		_xmin = std::vector<double>(nFunctions);
		_xmax = std::vector<double>(nFunctions);
		_deltax = std::vector<double>(nFunctions);
		_offset = std::vector<double>(nFunctions);
		_index = std::vector<int>(nFunctions);
		for (int i = 0; i < nFunctions; ++i) {
			_xmin[i] = umin;
			_xmax[i] = umax;
			SetLimits(i);
		}
	}
	Urysohn(const Urysohn& uri) {
		_xmin.clear();
		_xmin = std::vector<double>(uri._xmin.size());
		for (int i = 0; i < (int)uri._xmin.size(); ++i) {
			_xmin[i] = uri._xmin[i];
		}
		_xmax.clear();
		_xmax = std::vector<double>(uri._xmax.size());
		for (int i = 0; i < (int)uri._xmax.size(); ++i) {
			_xmax[i] = uri._xmax[i];
		}
		_deltax.clear();
		_deltax = std::vector<double>(uri._deltax.size());
		for (int i = 0; i < (int)uri._deltax.size(); ++i) {
			_deltax[i] = uri._deltax[i];
		}
		_offset.clear();
		_offset = std::vector<double>(uri._offset.size());
		for (int i = 0; i < (int)uri._offset.size(); ++i) {
			_offset[i] = uri._offset[i];
		}
		_index.clear();
		_index = std::vector<int>(uri._index.size());
		for (int i = 0; i < (int)uri._index.size(); ++i) {
			_index[i] = uri._index[i];
		}
		_model.clear();
		_model = std::vector<std::vector<double>>(uri._model.size());
		for (int i = 0; i < (int)uri._model.size(); ++i) {
			_model[i] = std::vector<double>(uri._model[i].size());
			for (int j = 0; j < (int)uri._model[i].size(); ++j) {
				_model[i][j] = uri._model[i][j];
			}
		}
	}
	double GetUrysohn(const std::vector<double>& inputs, std::vector<double>& derivatives, bool freezeModel) {
		double f = 0.0;
		for (int i = 0; i < (int)_model.size(); ++i) {
			f += GetFunction(i, inputs[i], derivatives[i], freezeModel);
		}
		return f / (double)_model.size();
	}
	double GetUrysohn(const std::vector<double>& inputs, bool freezeModel) {
		double f = 0.0;
		for (size_t i = 0; i < _model.size(); ++i) {
			f += GetFunction(i, inputs[i], freezeModel);
		}
		return f / (double)_model.size();
	}
	void Update(double delta) {
		for (int i = 0; i < (int)_model.size(); ++i) {
			Update(i, delta);
		}
	}
	void IncrementPoints() {
		for (int i = 0; i < (int)_model.size(); ++i) {
			IncrementPoints(i);
		}
	}
	void ShowData() {
		printf("Min, max, delta\n");
		for (int i = 0; i < (int)_xmin.size(); ++i) {
			printf("%7.4f ", _xmin[i]);
		}
		printf("\n");
		for (int i = 0; i < (int)_xmax.size(); ++i) {
			printf("%7.4f ", _xmax[i]);
		}
		printf("\n");
		for (int i = 0; i < (int)_deltax.size(); ++i) {
			printf("%7.4f ", _deltax[i]);
		}
		printf("\n\n");
		printf("Urysohn: rows = functions, cols = points\n");
		for (int i = 0; i < (int)_model.size(); ++i) {
			for (int j = 0; j < (int)_model[i].size(); ++j) {
				printf("% 10.4f ", _model[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	void Renormalize(double shift, double range) {
		for (int i = 0; i < (int)_model.size(); ++i) {
			for (int j = 0; j < (int)_model[i].size(); ++j) {
				_model[i][j] -= shift;
				_model[i][j] /= range;
			}
		}
	}
	void SetAllMinMax(double min, double max) {
		for (int i = 0; i < (int)_model.size(); ++i) {
			_xmin[i] = min;
			_xmax[i] = max;
			SetLimits(i);
		}
	}
	std::vector<double> GetAllMinValues() {
		std::vector<double> min;
		for (int i = 0; i < (int)_xmin.size(); ++i) {
			min.push_back(_xmin[i]);
		}
		return min;
	}
	std::vector<double> GetAllMaxValues() {
		std::vector<double> max;
		for (int i = 0; i < (int)_xmax.size(); ++i) {
			max.push_back(_xmax[i]);
		}
		return max;
	}
private:
	std::vector<double> _xmin;
	std::vector<double> _xmax;
	std::vector<double> _deltax;
	std::vector<double> _offset;
	std::vector<int> _index;
	std::vector<std::vector<double>> _model;
	//
	void SetLimits(int k) {
		_deltax[k] = (_xmax[k] - _xmin[k]) / (_model[k].size() - 1);
	}
	void IncrementPoints(int k) {
		int points = (int)_model[k].size() + 1;
		double deltax = (_xmax[k] - _xmin[k]) / (points - 1);
		std::vector<double> y(points);
		y[0] = _model[k][0];
		y[points - 1] = _model[k][_model[k].size() - 1];
		for (int i = 1; i < points - 1; ++i) {
			y[i] = GetFunction(k, _xmin[k] + i * deltax, true);
		}
		_deltax[k] = deltax;
		_model[k].clear();
		for (size_t i = 0; i < y.size(); i++)
		{
			_model[k].push_back(y[i]);
		}
	}
	void Update(int k, double residual) {
		double tmp = residual * _offset[k];
		_model[k][_index[k] + 1] += tmp;
		_model[k][_index[k]] += residual - tmp;
	}
	double GetFunction(int k, double x, double& derivative, bool freezeModel) {
		double f = GetFunction(k, x, freezeModel);
		derivative = GetDerivative(k);
		return f;
	}
	double GetDerivative(int k) {
		return (_model[k][_index[k] + 1] - _model[k][_index[k]]) / _deltax[k];
	}
	double GetFunction(int k, double x, bool freezeModel) {
		if (!freezeModel) {
			FixLimits(k, x);
		}
		else {
			if (x <= _xmin[k]) {
				_index[k] = 0;
				_offset[k] = 0.001;
				return _model[k][0];
			}
			if (x >= _xmax[k]) {
				_index[k] = (int)_model[k].size() - 2;
				_offset[k] = 0.999;
				return _model[k][_model[k].size() - 1];
			}
		}
		double R = (x - _xmin[k]) / _deltax[k];
		_index[k] = (int)(R);
		_offset[k] = R - _index[k];
		return _model[k][_index[k]] + (_model[k][_index[k] + 1] - _model[k][_index[k]]) * _offset[k];
	}
	void FixLimits(int k, double x) {
		if (x < _xmin[k]) {
			_xmin[k] = x;
			SetLimits(k);
		}
		if (x > _xmax[k]) {
			_xmax[k] = x;
			SetLimits(k);
		}
	}
};


