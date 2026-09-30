// Adapted from the BSFD prototype (code-for-BSFD, algorithmsClassic/reader.hpp).
// Changes: added read_from_rows() for in-memory tables; removed the unused
// experimental read_data_a/parse_line_compare paths (rand()-based, non-deterministic).
#ifndef READER_H
#define READER_H

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
class Reader {
public:
  std::vector<std::vector<int>> data;
  std::vector<std::vector<std::string>> dataset;
  std::vector<std::unordered_map<std::string, int>> value_map;
  std::vector<std::string> attributes;


  int nrow;
  int ncol;

  Reader() {
    data.reserve(1000000);
  }

  // In-memory loader: column names + raw string rows (values are hash-coded
  // per column exactly like the file-based path).
  void read_from_rows(const std::vector<std::string>& columns,
                      const std::vector<std::vector<std::string>>& rows) {
    attributes = columns;
    for (auto row : rows) {
      std::vector<int> coded;
      coded.reserve(row.size());
      for (size_t j = 0; j < row.size(); ++j) {
        coded.push_back(hash(row[j], static_cast<int>(j)));
      }
      data.emplace_back(std::move(coded));
    }
    nrow = data.size();
    ncol = value_map.size();
  }


  void read_data(std::string path) {
    std::ifstream in(path);

    // handle column names
    std::string line;
    std::getline(in, line);
    
    std::stringstream columns(line);
    std::string attribute;
    while(std::getline(columns, attribute, ','))
    {
        attributes.push_back(attribute);
    }

    for (std::string line; std::getline(in, line); ) {

      data.emplace_back();
      if (data.size() >= 1) {
        data[data.size() - 1].reserve(data[0].size());
      }

      parse_line(line, data[data.size() - 1]);
    }
    nrow = data.size();
    ncol = value_map.size();
  }

  void parse_line(std::string& line, std::vector<int>& row) {
    int left = 0;
    int right = 0;
    int col_idx = 0;
    while (right < line.length()) {
      if (line[right] == ',' && line[right + 1] != ' ') {
        // a new column
        auto value = line.substr(left, right - left);
        auto code = hash(value, col_idx);         //生成一个哈希码
        row.push_back(code);
        ++col_idx;
        left = right + 1;
      }
      ++right;
    }

    auto value = line.substr(left, right - left);
    auto code = hash(value, col_idx);    //生成hash code
    row.push_back(code);
  }

  int hash(std::string& value, int col_idx) {
    if (col_idx >= value_map.size()) {
      value_map.emplace_back(std::unordered_map<std::string, int>());
      value_map[value_map.size() - 1].reserve(100000);
    }
    auto iter = value_map[col_idx].find(value);
    if (iter != value_map[col_idx].end()) {
      return iter->second;
    } else {
      int code = value_map[col_idx].size();  //就是数值，这样每加一个元素,code+1，不会有重复
      value_map[col_idx][value] = code;
      return code;
    }
  }

  bool check_integrity() {
    bool flag = true;
    for (int i = 0; i < nrow; ++i) {
      if (data[i].size() != ncol) {
        std::cout << "Error!\n"
                  << "But row " << i << " has " << data[i].size() << " columns.\n";
        flag = false;
      }
    }
    std::cout << "Total Rows: " << nrow << "\n"
              << "Total Cols: " << ncol << "\n";
    return flag;
  }
};

#endif // READER_H
