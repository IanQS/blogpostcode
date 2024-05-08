//
// Created by ian on 9/29/20.
//

#include <string>
#include <vector>
#include <complex>
#include <tuple>
#include <map>
#ifndef PALISADE_TUTORIAL_UTILS_CSV_READER_H_
#define PALISADE_TUTORIAL_UTILS_CSV_READER_H_

using FeatureMatrixComplex = std::vector<std::vector<std::complex<double>>>;
using LabelVector = std::vector<int>;
using FeatureNameMap = std::map<std::string, int>;


FeatureMatrixComplex readFeatures(const std::string &dataFile);
std::tuple<LabelVector, FeatureNameMap> readLabels(const std::string &dataFIle);

#endif //PALISADE_TUTORIAL_UTILS_CSV_READER_H_
