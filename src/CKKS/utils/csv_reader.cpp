//
// Created by ian on 9/29/20.
//

#include "csv_reader.h"
#include <fstream>
#include <complex>
#include <vector>
#include <iostream>


FeatureMatrixComplex readFeatures(const std::string &dataFile){
  FeatureMatrixComplex container;
  std::ifstream file(dataFile);

  std::string row;
  if (file.is_open()){
    while (std::getline(file, row)){
      std::stringstream ss(row);
      std::vector<std::complex<double>> featureRow;
      std::string scalar;
      while (std::getline(ss, scalar, ',')) {
        featureRow.emplace_back(std::stod(scalar), 0.0);
      }
      container.emplace_back(featureRow);
    }
  }
  return container;
}

std::tuple<LabelVector,FeatureNameMap>  readLabels(const std::string &dataFile){
  LabelVector container;
  std::ifstream file(dataFile);
  std::map<std::string, int> nameMap;

  std::string flower_name_label;
  int label_count = -1;
  if (file.is_open()){
    while (std::getline(file, flower_name_label)){
      if (!(flower_name_label.empty()) && (nameMap.find(flower_name_label) == nameMap.end())){
        // Did not exist, so we scan to the end
        label_count += 1;
        nameMap.emplace(flower_name_label, label_count);
      }
      container.push_back(nameMap.find(flower_name_label)->second);
    }
  }

  return std::make_tuple(container, nameMap);
}
