//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// he or she is under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://arxiv.org/abs/2305.08194

//Compiles in Linux, here is the Makefile
//# Compiler and flags
//CXX = g++
//CXXFLAGS = -O2 - std = c++17 - Wall
//
//# Target name(final executable)
//TARGET = KANKAN - 6
//
//# Source files
//SRCS = KANKAN - 6.cpp
//
//# Object files
//OBJS = $(SRCS:.cpp = .o)
//
//# Default rule
//$(TARGET) : $(OBJS)
//$(CXX) $(CXXFLAGS) - o $@ $(OBJS)
//
//# Compile.cpp to.o
//% .o: % .cpp
//$(CXX) $(CXXFLAGS) - c $ < -o $@
//
//	# Clean rule
//	clean :
//rm - f $(OBJS) $(TARGET)

#include <iostream>
#include <cmath>
#include "Helper.h"
#include "Urysohn.h"
#include "Layer.h"

//We don't need 4 layers here, it is only a demo how to make 4 layers
void AreasOfTriangles() {
	int nFeatures = 6;
	int nTargets = 1;
	int nTrainingRecords = 10000;
	int nValidationRecords = 2000;
	double min = 0.0;
	double max = 1.0;
	auto features_training = Helper::MakeRandomMatrixForTriangles(nTrainingRecords, nFeatures, min, max);
	auto features_validation = Helper::MakeRandomMatrixForTriangles(nValidationRecords, nFeatures, min, max);
	auto targets_training = Helper::ComputeAreasOfTriangles(features_training);
	auto targets_validation = Helper::ComputeAreasOfTriangles(features_validation);

	//data is ready, we start training
	clock_t start_application = clock();
	clock_t current_time = clock();

	std::vector<double> argmin;
	std::vector<double> argmax;
	for (int i = 0; i < nFeatures; ++i) {
		argmin.push_back(min);
		argmax.push_back(max);
	}

	double targetMin = Helper::MinV(targets_training);
	double targetMax = Helper::MaxV(targets_training);

	int nU0 = 50;
	int nU1 = 10;
	int nU2 = 4;
	int nU3 = nTargets;
	double alpha = 0.005;

	auto layer0 = std::make_unique<Layer>(nU0, argmin, argmax, targetMin, targetMax, 2);
	auto layer1 = std::make_unique<Layer>(nU1, nU0, targetMin, targetMax, 12);
	auto layer2 = std::make_unique<Layer>(nU2, nU1, targetMin, targetMax, 12);
	auto layer3 = std::make_unique<Layer>(nU3, nU2, targetMin, targetMax, 22);

	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> models2(nU2);
	std::vector<double> models3(nU3);

	std::vector<double> deltas3(nU3);
	std::vector<double> deltas2(nU2);
	std::vector<double> deltas1(nU1);
	std::vector<double> deltas0(nU0);

	auto actual0 = std::make_unique<double[]>(nValidationRecords);
	auto computed0 = std::make_unique<double[]>(nValidationRecords);

	printf("Targets are areas of random triangles, %d training records\n", nTrainingRecords);
	for (int epoch = 0; epoch < 128; ++epoch) {
		for (int i = 0; i < nTrainingRecords; ++i) {		
			layer0->Input2Output(features_training[i], models0, false);
			layer1->Input2Output(models0, models1, false, true);
			layer2->Input2Output(models1, models2, false, true);
			layer3->Input2Output(models2, models3, false, true);

			deltas3[0] = (targets_training[i] - models3[0]) * alpha;
			
			layer3->ComputeDeltas(deltas3, deltas2);
			layer2->ComputeDeltas(deltas2, deltas1);
			layer1->ComputeDeltas(deltas1, deltas0);
			
			layer3->Update(deltas3);
			layer2->Update(deltas2);
			layer1->Update(deltas1);
			layer0->Update(deltas0);
		}

		double error = 0.0;
		for (int i = 0; i < nValidationRecords; ++i) {
			layer0->Input2Output(features_validation[i], models0);
			layer1->Input2Output(models0, models1);
			layer2->Input2Output(models1, models2);
			layer3->Input2Output(models2, models3);

			double err = targets_validation[i] - models3[0];
			error += err * err;

			actual0[i] = targets_validation[i];
			computed0[i] = models3[0];
		}

		//pearsons for correlated targets
		double p1 = Helper::Pearson(computed0, actual0, nValidationRecords);

		//mean error
		error /= nTargets;
		error /= nValidationRecords;
		error = sqrt(error);
		current_time = clock();
		printf("Epoch %d, RMSE %f, Pearson: %f, time %2.3f\n", epoch, error, p1,
			(double)(current_time - start_application) / CLOCKS_PER_SEC);

		if (p1 > 0.985) break;
	}
	printf("\n");
}

void Medians() {
	int nTrainingRecords = 10000;
	int nValidationRecords = 2000;
	int nFeatures = 6;
	int nTargets = 3;
	double min = 0.0;
	double max = 1.0;
	auto features_training = Helper::GenerateInputsMedians(nTrainingRecords, nFeatures, min, max);
	auto features_validation = Helper::GenerateInputsMedians(nValidationRecords, nFeatures, min, max);
	auto targets_training = Helper::ComputeTargetsMedians(features_training);
	auto targets_validation = Helper::ComputeTargetsMedians(features_validation);

	//data is ready, we start training
	clock_t start_application = clock();
	clock_t current_time = clock();

	std::vector<double> argmin;
	std::vector<double> argmax;
	for (int i = 0; i < nFeatures; ++i) {
		argmin.push_back(min);
		argmax.push_back(max);
	}

	double targetMin = DBL_MAX;
	double targetMax = -DBL_MAX;
	for (int i = 0; i < nTrainingRecords; ++i) {
		for (int j = 0; j < nTargets; ++j) {
			if (targets_training[i][j] < targetMin) targetMin = targets_training[i][j];
			if (targets_training[i][j] > targetMax) targetMax = targets_training[i][j];
		}
	}

	int nU0 = 20;
	int nU1 = 10;
	int nU2 = 4;
	int nU3 = nTargets;
	double alpha = 0.005;

	auto layer0 = std::make_unique<Layer>(nU0, argmin, argmax, targetMin, targetMax, 2);
	auto layer1 = std::make_unique<Layer>(nU1, nU0, targetMin, targetMax, 12);
	auto layer2 = std::make_unique<Layer>(nU2, nU1, targetMin, targetMax, 12);
	auto layer3 = std::make_unique<Layer>(nU3, nU2, targetMin, targetMax, 22);

	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> models2(nU2);
	std::vector<double> models3(nU3);

	std::vector<double> deltas3(nU3);
	std::vector<double> deltas2(nU2);
	std::vector<double> deltas1(nU1);
	std::vector<double> deltas0(nU0);

	auto actual0 = std::make_unique<double[]>(nValidationRecords);
	auto actual1 = std::make_unique<double[]>(nValidationRecords);
	auto actual2 = std::make_unique<double[]>(nValidationRecords);

	auto computed0 = std::make_unique<double[]>(nValidationRecords);
	auto computed1 = std::make_unique<double[]>(nValidationRecords);
	auto computed2 = std::make_unique<double[]>(nValidationRecords);

	printf("Targets are medians of random triangles, %d training records\n", nTrainingRecords);
	for (int epoch = 0; epoch < 128; ++epoch) {
		for (int i = 0; i < nTrainingRecords; ++i) {
			layer0->Input2Output(features_training[i], models0, false);
			layer1->Input2Output(models0, models1, false, true);
			layer2->Input2Output(models1, models2, false, true);
			layer3->Input2Output(models2, models3, false, true);

			for (int j = 0; j < nTargets; ++j) {
				deltas3[j] = (targets_training[i][j] - models3[j]) * alpha;
			}

			layer3->ComputeDeltas(deltas3, deltas2);
			layer2->ComputeDeltas(deltas2, deltas1);
			layer1->ComputeDeltas(deltas1, deltas0);

			layer3->Update(deltas3);
			layer2->Update(deltas2);
			layer1->Update(deltas1);
			layer0->Update(deltas0);
		}

		double error = 0.0;
		for (int i = 0; i < nValidationRecords; ++i) {
			layer0->Input2Output(features_validation[i], models0);
			layer1->Input2Output(models0, models1);
			layer2->Input2Output(models1, models2);
			layer3->Input2Output(models2, models3);

			for (int j = 0; j < nTargets; ++j) {
				double err = targets_validation[i][j] - models3[j];
				error += err * err;
			}

			actual0[i] = targets_validation[i][0];
			actual1[i] = targets_validation[i][1];
			actual2[i] = targets_validation[i][2];

			computed0[i] = models3[0];
			computed1[i] = models3[1];
			computed2[i] = models3[2];
		}

		//pearsons for correlated targets
		double p1 = Helper::Pearson(computed0, actual0, nValidationRecords);
		double p2 = Helper::Pearson(computed1, actual1, nValidationRecords);
		double p3 = Helper::Pearson(computed2, actual2, nValidationRecords);

		//mean error
		error /= nTargets;
		error /= nValidationRecords;
		error = sqrt(error);
		current_time = clock();
		printf("Epoch %d, RMSE %f, Pearsons: %f, %f, %f, time %2.3f\n", epoch, error, p1, p2, p3,
			(double)(current_time - start_application) / CLOCKS_PER_SEC);

		if (p1 > 0.985 && p2 > 0.985 && p3 > 0.985) break;
	}
	printf("\n");
}

//Demo how to use Layers without KANKAN wrapper
void Det_4_4() {
	int nTrainingRecords = 100000;
	int nValidationRecords = 20000;
	int nMatrixSize = 4;
	int nFeatures = nMatrixSize * nMatrixSize;
	double min = 0.0;
	double max = 10.0;
	auto features_training = Helper::GenerateInput(nTrainingRecords, nFeatures, min, max);
	auto features_validation = Helper::GenerateInput(nValidationRecords, nFeatures, min, max);
	auto targets_training = Helper::ComputeDeterminantTarget(features_training, nMatrixSize);
	auto targets_validation = Helper::ComputeDeterminantTarget(features_validation, nMatrixSize);

	clock_t start_application = clock();
	clock_t current_time = clock();

	//find limits
	std::vector<double> argmin;
	std::vector<double> argmax;
	for (int i = 0; i < nFeatures; ++i) {
		argmin.push_back(min);
		argmax.push_back(max);
	}

	double targetMin = Helper::MinV(targets_training);
	double targetMax = Helper::MaxV(targets_training);

	//configuration
	int nU0 = 64;
	int nU1 = 1;
	double alpha = 0.1;

	//instantiation of layers
	auto layer0 = std::make_unique<Layer>(nU0, argmin, argmax, targetMin, targetMax, 3);
	auto layer1 = std::make_unique<Layer>(nU1, nU0, targetMin, targetMax, 30);

	//auxiliary data buffers for a quick moving data between methods
	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);

	std::vector<double> deltas1(nU1);
	std::vector<double> deltas0(nU0);

	//std::vector<std::vector<double>> derivatives1(nU1, std::vector<double>(nU0, 0.0));
	auto actual_validation = std::make_unique<double[]>(nValidationRecords);

	//training
	printf("Targets are determinants of random 4 * 4 matrices, %d training records\n", nTrainingRecords);
	for (int epoch = 0; epoch < 128; ++epoch) {
		//test of incrementing of the number of linear segments
		if (3 == epoch) {
			layer0->IncrementPoins();
			layer1->IncrementPoins();
		}
		for (int i = 0; i < nTrainingRecords; ++i) {
			//forward feeding by two layers
			layer0->Input2Output(features_training[i], models0, false);
			layer1->Input2Output(models0, models1, false, true);

			//computing residual error
			deltas1[0] = (targets_training[i] - models1[0]) * alpha;

			//back propagation
			layer1->ComputeDeltas(deltas1, deltas0);

			//updating of two layers
			layer1->Update(deltas1);
			layer0->Update(deltas0);
		}

		//validation at the end of each epoch
		double error = 0.0;
		for (int i = 0; i < nValidationRecords; ++i) {
			layer0->Input2Output(features_validation[i], models0);
			layer1->Input2Output(models0, models1);
			actual_validation[i] = models1[0];
			error += (targets_validation[i] - models1[0]) * (targets_validation[i] - models1[0]);
		}
		double pearson = Helper::Pearson(targets_validation, actual_validation, nValidationRecords);
		error /= nValidationRecords;
		error = sqrt(error);
		error /= (targetMax - targetMin);
		current_time = clock();
		printf("Epoch %d, current relative error %f, pearson %f, time %2.3f\n", epoch, error, pearson, (double)(current_time - start_application) / CLOCKS_PER_SEC);

		if (pearson > 0.97) break;
	}

	//test of copy constructor
	auto copy_layer0 = std::make_unique<Layer>(*layer0);
	auto copy_layer1 = std::make_unique<Layer>(*layer1);

	//test of renormalization
	auto xmin = copy_layer1->GetAllMinValues(0);
	auto xmax = copy_layer1->GetAllMaxValues(0);
	copy_layer0->RenormalizeAllU(xmin, xmax, targetMin, targetMax);
	copy_layer1->SetMinMaxAllU(targetMin, targetMax, 0);

	double error2 = 0.0;
	for (int i = 0; i < nValidationRecords; ++i) {
		layer0->Input2Output(features_validation[i], models0);
		layer1->Input2Output(models0, models1);
		actual_validation[i] = models1[0];
		error2 += (targets_validation[i] - models1[0]) * (targets_validation[i] - models1[0]);
	}
	double pearson = Helper::Pearson(targets_validation, actual_validation, nValidationRecords);
	error2 /= nValidationRecords;
	error2 = sqrt(error2);
	error2 /= (targetMax - targetMin);
	current_time = clock();
	printf("Relative error of copy %f, pearson %f, time %2.3f\n\n", error2, pearson, (double)(current_time - start_application) / CLOCKS_PER_SEC);
}

//Here I show how to use Layers directly without KANKAN wrapper
void Tetrahedron() {
	const int nTrainingRecords = 500000;
	const int nValidationRecords = 50000;
	const int nFeatures = 12;
	const int nTargets = 4;
	const double min = 0.0;
	const double max = 10.0;
	auto features_training = Helper::MakeRandomMatrix(nTrainingRecords, nFeatures, min, max);
	auto features_validation = Helper::MakeRandomMatrix(nValidationRecords, nFeatures, min, max);
	auto targets_training = Helper::ComputeTargetMatrix(features_training);
	auto targets_validation = Helper::ComputeTargetMatrix(features_validation);

	//data is ready, we start training
	clock_t start_application = clock();
	clock_t current_time = clock();

	std::vector<double> argmin;
	std::vector<double> argmax;
	for (int i = 0; i < nFeatures; ++i) {
		argmin.push_back(min);
		argmax.push_back(max);
	}

	double targetMin = DBL_MAX;
	double targetMax = -DBL_MAX;
	for (int i = 0; i < nTrainingRecords; ++i) {
		for (int j = 0; j < nTargets; ++j) {
			if (targets_training[i][j] < targetMin) targetMin = targets_training[i][j];
			if (targets_training[i][j] > targetMax) targetMax = targets_training[i][j];
		}
	}

	int nU0 = 60;
	int nU1 = 10;
	int nU2 = nTargets;
	double alpha = 0.05;

	auto layer0 = std::make_unique<Layer>(nU0, argmin, argmax, targetMin, targetMax, 2);
	auto layer1 = std::make_unique<Layer>(nU1, nU0, targetMin, targetMax, 12);
	auto layer2 = std::make_unique<Layer>(nU2, nU1, targetMin, targetMax, 22);

	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> models2(nU2);

	std::vector<double> deltas2(nU2);
	std::vector<double> deltas1(nU1);
	std::vector<double> deltas0(nU0);

	auto actual0 = std::make_unique<double[]>(nValidationRecords);
	auto actual1 = std::make_unique<double[]>(nValidationRecords);
	auto actual2 = std::make_unique<double[]>(nValidationRecords);
	auto actual3 = std::make_unique<double[]>(nValidationRecords);

	auto computed0 = std::make_unique<double[]>(nValidationRecords);
	auto computed1 = std::make_unique<double[]>(nValidationRecords);
	auto computed2 = std::make_unique<double[]>(nValidationRecords);
	auto computed3 = std::make_unique<double[]>(nValidationRecords);

	printf("Targets are areas of faces of random tetrahedrons, %d\n", nTrainingRecords);
	for (int epoch = 0; epoch < 128; ++epoch) {
		for (int i = 0; i < nTrainingRecords; ++i) {
			layer0->Input2Output(features_training[i], models0, false);
			layer1->Input2Output(models0, models1, false, true);
			layer2->Input2Output(models1, models2, false, true);

			for (int j = 0; j < nTargets; ++j) {
				deltas2[j] = (targets_training[i][j] - models2[j]) * alpha;
			}

			layer2->ComputeDeltas(deltas2, deltas1);
			layer1->ComputeDeltas(deltas1, deltas0);

			layer2->Update(deltas2);
			layer1->Update(deltas1);
			layer0->Update(deltas0);
		}

		double error = 0.0;
		for (int i = 0; i < nValidationRecords; ++i) {
			layer0->Input2Output(features_validation[i], models0);
			layer1->Input2Output(models0, models1);
			layer2->Input2Output(models1, models2);

			for (int j = 0; j < nTargets; ++j) {
				double err = targets_validation[i][j] - models2[j];
				error += err * err;
			}

			actual0[i] = targets_validation[i][0];
			actual1[i] = targets_validation[i][1];
			actual2[i] = targets_validation[i][2];
			actual3[i] = targets_validation[i][3];

			computed0[i] = models2[0];
			computed1[i] = models2[1];
			computed2[i] = models2[2];
			computed3[i] = models2[3];
		}
		double p1 = Helper::Pearson(computed0, actual0, nValidationRecords);
		double p2 = Helper::Pearson(computed1, actual1, nValidationRecords);
		double p3 = Helper::Pearson(computed2, actual2, nValidationRecords);
		double p4 = Helper::Pearson(computed3, actual3, nValidationRecords);

		error /= nTargets;
		error /= nValidationRecords;
		error = sqrt(error);
		current_time = clock();
		printf("Epoch %d, RMSE %f, Pearsons: %f %f %f %f, time %2.3f\n", epoch, error, p1, p2, p3, p4,
			(double)(current_time - start_application) / CLOCKS_PER_SEC);

		if (p1 > 0.975 && p2 > 0.975 && p3 > 0.975 && p4 > 0.975) break;
	}
	printf("\n");
}

int main() {
	srand((unsigned int)time(NULL));
	
	//These are unit tests

	//The areas of the faces of tetrahedron given by random vertices.
	Tetrahedron();

	//Deternminants of random matrices of 4 by 4.
	Det_4_4();

	//Related targets, the lengths of medians of random triangles.
	Medians();
	
	//Areas of random triangles.
	AreasOfTriangles();
}

