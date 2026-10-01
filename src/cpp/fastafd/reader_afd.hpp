// In-memory dataset transformation for FastAFD (FDM, ICDE 2024 section V):
// representative-tuple reservoir + tuple-pair binary comparison matrix.
//
// Adapted from the prototype's reader.hpp: file I/O removed (rows arrive in
// memory), rand() replaced by a seeded std::mt19937, and the MAX_SIZE /
// MIN_COMPARE / MIN_MATRIX macros are constructor parameters.
#ifndef FASTAFD_READER_H_
#define FASTAFD_READER_H_

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

namespace fastafd {

class ReaderAFD {
public:
  // Sampled binary comparison matrix: each row is the per-attribute equality
  // vector of a pair of representative tuples.
  std::vector<std::vector<int>> data;
  std::vector<std::vector<std::string>> dataset;  // representative tuple pool
  std::vector<std::string> attributes;

  int nrow = 0;   // effective row count (pool size + accepted comparisons)
  int ncol = 0;

  const int max_size;      // reservoir size (paper: 200)
  const int min_compare;   // min matching attributes to keep a comparison (3)
  const int min_matrix;    // min comparisons to trigger reservoir update (2)
  std::mt19937 rng;

  ReaderAFD(uint64_t seed = 42, int max_size = 200, int min_compare = 3, int min_matrix = 2)
      : max_size(max_size), min_compare(min_compare), min_matrix(min_matrix), rng(seed) {
    data.reserve(1000000);
    dataset.reserve(max_size + 1);
  }

  // In-memory variant of the prototype's read_data_a().
  void read_from_rows(const std::vector<std::string>& columns,
                      const std::vector<std::vector<std::string>>& rows) {
    attributes = columns;
    ncol = static_cast<int>(attributes.size());
    if (ncol > static_cast<int>(bitmap.size())) {
      throw std::invalid_argument("FastAFD supports at most 40 columns per call");
    }
    int countk = max_size;
    std::vector<std::vector<int>> tmp_m;

    for (const auto& row : rows) {
      if (static_cast<int>(dataset.size()) < max_size) {
        dataset.push_back(row);
      } else {
        tmp_m.clear();
        for (int i = 0; i < max_size; i++) {
          int match = 0;
          std::vector<int> tmp_v;
          tmp_v.reserve(ncol);
          for (int j = 0; j < ncol; j++) {
            int same = dataset[i][j] == row[j] ? 1 : 0;
            tmp_v.push_back(same);
            match += same;
          }
          if (match >= min_compare) {
            tmp_m.push_back(std::move(tmp_v));
          }
        }
        if (static_cast<int>(tmp_m.size()) >= min_matrix) {
          countk++;
          for (auto& v : tmp_m) {
            data.push_back(std::move(v));
          }
          int rand_int = static_cast<int>(rng() % (countk - 1)) + 1;
          if (rand_int < max_size) {
            dataset[rand_int - 1] = row;
          }
        }
      }
    }
    nrow = countk;
    if (static_cast<int>(dataset.size()) < max_size) {
      // Fewer rows than the reservoir: compare all pairs directly.
      // (The prototype produced no matrix at all in this case; the all-pairs
      // matrix keeps small tables usable.)
      data = build_full_matrix_from_pool();
      nrow = static_cast<int>(dataset.size());
    }
  }

private:
  std::vector<std::vector<int>> build_full_matrix_from_pool() {
    std::vector<std::vector<int>> matrix;
    for (size_t i = 0; i < dataset.size(); ++i) {
      for (size_t j = i + 1; j < dataset.size(); ++j) {
        std::vector<int> cmp;
        cmp.reserve(ncol);
        int match = 0;
        for (int col = 0; col < ncol; ++col) {
          int same = dataset[i][col] == dataset[j][col] ? 1 : 0;
          cmp.push_back(same);
          match += same;
        }
        if (match >= min_compare) matrix.push_back(std::move(cmp));
      }
    }
    return matrix;
  }
};

}  // namespace fastafd

#endif  // FASTAFD_READER_H_
