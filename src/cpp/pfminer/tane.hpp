// Adapted from the BSFD prototype (code-for-BSFD, algorithmsClassic/TANE.hpp).
// Changes: SUPPORT/CONFIDENCE are runtime members instead of #defines, and an
// in-memory loader (set_data) replaces file-only input.
#ifndef TANE_H
#define TANE_H

#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <utility>
#include "../common/reader.hpp"
typedef long long Colbit;
using namespace std;

//计算输入参数的2次幂
inline long long encode(int item) {
  return 1LL << item;
}

inline long long encode(std::vector<int>& items) {
  long long code = 0;
  for (int i = 0; i < items.size(); ++i) {
    code |= (1LL << items[i]);
  }
  return code;
}

inline std::vector<int> decode_to_vector(long long code) {
  std::vector<int> items;
  for (int i = 0; code != 0; ++i) {
    if (code % 2 == 1) {
      items.push_back(i);
    }
    code /= 2;
  }
  return items;
}

inline std::unordered_set<int> decode_to_set(long long code) {
  std::unordered_set<int> items;
  for (int i = 0; code != 0; ++i) {
    if (code % 2 == 1) {
      items.insert(i);
    }
    code /= 2;
  }
  return items;
}

inline bool contains(long long code, int item) {
  int s = encode(item);
  return (s & code);
}

inline long long exclude_set(long long code1, long long code2) {
  return (code1 ^ code2);
}

inline long long exclude_item(long long code, int item) {
  if (contains(code, item)) {
    return (code - encode(item));
  }
  return code;
}

inline long long merge_set(long long code1, long long code2) {
  return (code1 | code2);
}

inline long long merge_item(long long code, int item) {
  if (contains(code, item)) {
    return code;
  }
  return (code | (1LL << item));
}

inline long long intersect(long long code1, long long code2) {
  return (code1 & code2);
}

// how many bits are different
inline int difference(long long code1, long long code2) {
  int ret = 0;
  long long code = code1 ^ code2;
  while (code) {
    code = code & (code - 1);
    ++ret;
  }
  return ret;
}

class TANE {
public:
  //二维的
  using Partition = vector<std::vector<int>>;    

  std::vector<std::vector<int>> data;
  std::vector<int> T;
  std::unordered_map<long long, int> C;
  std::vector<long long> L;
  std::unordered_map<long long, Partition> set_part_map;   //整数到partiton的映射
  std::vector<std::pair<long long, int>> FD;

  std::unordered_map<long long, std::pair<long long, long long>> parents;
  std::unordered_map<long long, int> eX;

  std::vector<std::string> attributes;
  Partition counter;
  Partition S;

  int nrow;
  int ncol;
  long long full_set;
  // Mining thresholds (paper defaults; were #defines in the prototype).
  double support_threshold = 0.10;
  double confidence_threshold = 0.90;
//默认构造函数
  TANE () = default;

  // Load an in-memory table (columns + hash-coded rows) prepared by Reader.
  inline void set_data(Reader& r) {
    data = std::move(r.data);
    nrow = r.nrow;
    ncol = r.ncol;
    T.resize(nrow);
    attributes = std::move(r.attributes);
    full_set = 0;
    for (int i = 0; i < ncol; ++i) {
      full_set = (full_set << 1) + 1;
    }
    set_part_map.reserve(15000);
  }
  inline void run() {
    
    L.clear();
    for (int i = 0; i < ncol; ++i) {
      //把每一列编码成整数？好计算？
      L.emplace_back(encode(i));
    }
    
    set_part_map.clear();
    for (int i = 0; i < ncol; ++i) {
      set_part_map[L[i]] = std::move(one_column_partition(i));
    }

    C[0] = full_set;
    while (!L.empty()) {    //层次搜索空间
      compute_dependencies();
      // cout<<"compute_dependencies~"<<endl;
      prune();
      // cout<<"prune~"<<endl;
      generate_next_level();
      // cout<<"generate_next_level~"<<endl;
    }
    
  }

  inline void read_data(std::string& path) {
    Reader r;
    r.read_data(path);
    cout<<"nrow:"<<r.nrow<<" ncol:"<<r.ncol<<"\n";
    set_data(r);
  }

  inline void generate_next_level() {
    std::vector<long long> new_level;
    new_level.reserve(L.size() * L.size() / 2);
    std::unordered_set<int> visited;
    //在同一层两两遍历，不考虑自己和自己
    for (int i = 0; i < L.size(); ++i) {
      for (int j = i + 1; j < L.size(); ++j) {
        
        auto s1 = L[i];
        auto s2 = L[j];
        //如果两个集合有两个不同的属性，则合并？
        if (difference(s1, s2) == 2) {
          long long merged = merge_set(s1, s2);
          if (visited.find(merged) == visited.end()) {
            visited.insert(merged);
            new_level.push_back(merged);
            //merged是由s1,s2合并儿来
            parents[merged] = std::make_pair(s1, s2);
          }
        }
      }
    }
    L = std::move(new_level);
  }

//单值划分
  inline Partition one_column_partition(int col) {
    Partition tmp;
    //逐行进行分割
    for (int ridx = 0; ridx < data.size(); ++ridx) {
      auto cat = data[ridx][col];
      //tmp[2]={2,4,6},表示的是当前列，值为2的行索引是{2，4，6}
      if (cat >= tmp.size()) {
        tmp.emplace_back();
      }
      tmp[cat].push_back(ridx);
    }
    Partition ret;
    for (int i = 0; i < tmp.size(); ++i) {
      if (tmp[i].size() > 1) {
        ret.emplace_back(tmp[i]);
      }
    }
    return ret;
  }

//合并两个分区
  inline void multiply_partitions(Partition& lhs, Partition& rhs, Partition& buf) {
    buf.reserve(150);
    init_T();
    //
    for (int cidx = 0; cidx < lhs.size(); ++cidx) {
      auto& p = lhs[cidx];
      for (int i = 0; i < p.size(); ++i) {
        T[p[i]] = cidx;
      }
    }
    while (S.size() < lhs.size()) {
      S.emplace_back();
    }
    for (int cidx = 0; cidx < rhs.size(); ++cidx) {
      auto& p = rhs[cidx];
      for (int i = 0; i < p.size(); ++i) {
        auto ridx = p[i];
        if (T[ridx] != -1) {
          S[T[ridx]].push_back(ridx);
        }
      }
      for (int i = 0; i < p.size(); ++i) {
        auto ridx = p[i];
        if (T[ridx] != -1) {
          if (S[T[ridx]].size() > 1) {
            buf.emplace_back();
            buf[buf.size() - 1] = S[T[ridx]];
          }
          S[T[ridx]].clear();
        }
      }
    }
  }

  std::vector<std::pair<long long, int>>& getFD() {
		return FD;
	}

  inline void compute_dependencies() {
    for (int i = 0; i < L.size(); ++i) {
      auto X = L[i];
      C[X] = full_set;
      //表示有哪些列？
      auto Xset = decode_to_vector(X);

      for (int j = 0; j < Xset.size(); ++j) {
        auto A = Xset[j];
        int X_without_A = exclude_item(X, A);
        C[X] = intersect(C[X], C[X_without_A]);
      }
    }
    // cout<<"compute C plus"<<endl;
    //C[X]才是rhs候选集
    for (int i = 0; i < L.size(); ++i) {
      //每一层的L提供的是lhs
      auto& X = L[i];
      auto candidates = decode_to_vector(intersect(X, C[X]));
      // cout<<candidates.size()<<endl;
      for (int j = 0; j < candidates.size(); ++j) {
        auto A = candidates[j];
        // cout<<"********"<<j<<endl;
        if (isValid(X, A)) {
          // cout<<"isValid"<<endl;
          FD.emplace_back(exclude_item(X, A), A);
          // cout<<"FD.emplace_back"<<endl;
          C[X] = exclude_item(C[X], A);
          auto CX_set = decode_to_vector(C[X]);
          for (int k = 0; k < CX_set.size(); ++k) {
            auto B = CX_set[k];
            if (!contains(X, B)) {
              C[X] = exclude_item(C[X], B);
            }
          }
        }
        // cout<<"======="<<j<<endl;
      }
      // cout<<"=========="<<endl;
    }
    // cout<<"compute FD"<<endl;
  }

  double calculate_error(Partition& lhs, Partition& l_rhs) {
      double l_error = 0.000;
      double lr_error = 0.000;
      double error = 0.000;
      for (int cidx = 0; cidx < lhs.size(); cidx++) {
          auto& p = lhs[cidx];
          l_error = l_error + (std::pow(p.size(), 2) - p.size());
      }
      for (int cidx = 0; cidx < l_rhs.size(); cidx++) {
          auto& p = l_rhs[cidx];
          lr_error = lr_error + (std::pow(p.size(), 2) - p.size());
      }
      error = (l_error - lr_error) / (std::pow(nrow, 2) - nrow);
      // std::cout << "true_error:" << error << "\n ";
      return error;
  }
  double calculate_confidence(Partition& lhs, Partition& l_rhs) {
      double l_conf = 0.000;
      double lr_conf = 0.000;
      double confidence = 0.000;
      for (int cidx = 0; cidx < lhs.size(); cidx++) {
          auto& p = lhs[cidx];
          l_conf = l_conf + (std::pow(p.size(), 2) - p.size());
      }
      for (int cidx = 0; cidx < l_rhs.size(); cidx++) {
          auto& p = l_rhs[cidx];
          lr_conf = lr_conf + (std::pow(p.size(), 2) - p.size());
      }
      confidence = lr_conf / l_conf;
      // std::cout << "true_error:" << error << "\n ";
      return confidence;
  }
//对一个划分里的元组进行支持度的计算？
  double calculate_support(Partition& l_rhs) {
      double sup = 0.000;
      for (int cidx = 0; cidx < l_rhs.size(); cidx++) {
          auto& p = l_rhs[cidx];
          //直接用分区的大小进行计算  ((n^2 - n) / 2) *2
          sup = sup + (std::pow(p.size(), 2) - p.size());
      }
      return sup / (std::pow(nrow, 2) - nrow);
      // return sup / nrow;
  }

  std::pair<double, double> calculate_pertuple_prob(Partition& lhs, Partition& l_rhs) {
        double numerator = 0.000;
        double denominator = 0.000;
        std::pair<double, double> criterions = std::make_pair(0.000, 0.000);
        for(const auto& VX : lhs){
            std::unordered_map<int, int> VA_count;
            int max_count_VA = 0;
            for (const auto& VXA : l_rhs) {
            // 检查 VX 和 VXA 是否有交集（是否属于同一个 VX）
                if (std::any_of(VX.begin(), VX.end(), [&](int tuple_id) {
                        return std::find(VXA.begin(), VXA.end(), tuple_id) != VXA.end();
                    })) {
                    // VXA 中的 VA 是 VXA 的最后一个元素
                    // int VA = VXA.back();
                    // VA_count[VA]++;
                    int cur_VXA = VXA.size();
                    max_count_VA = std::max(max_count_VA, cur_VXA);
                }
                
                // for(const auto& [VA, count] : VA_count){
                    
                // }
            
            }
            numerator += max_count_VA;
            denominator += VX.size();
        }
        
        criterions.first = numerator / denominator;
        criterions.second = numerator/ nrow;
        return criterions;
    }
  inline bool isValid(long long bigX, int A) {
    double t_error;
    double num;
    double conf;
    std::pair<double, double> criterions;
    auto X = exclude_item(bigX, A);
    if (!X) {
      return false;
    }
    compute_partition_on_demand(X);
    // cout<<" compute_partition_on_X"<<endl;
    // cout<<"bigX:"<<bigX<<endl;
    // cout<<"A:"<<A<<endl;
    // cout<<"X:"<<X<<endl;
    compute_partition_on_demand(bigX);
    // cout<<" compute_partition_on_bigX"<<endl;
    t_error = calculate_error(set_part_map[X], set_part_map[bigX]);
    conf = calculate_confidence(set_part_map[X], set_part_map[bigX]);
    num = calculate_support(set_part_map[bigX]);
//    criterions = calculate_pertuple_prob(set_part_map[X], set_part_map[bigX]);
//     std::cout << "conf:" << criterions.first<<"supp:"<<criterions.second << "\n ";
      if (num >= support_threshold && conf >= confidence_threshold) {
      return true;
    }
    return false;
  }

  inline void init_T() {
    for (auto iter = T.begin(); iter != T.end(); ++iter) {
      (*iter) = -1;
    }
  }

//剪枝就是对这一层生成的候选集删去无用的东西
  inline void prune() {
    double num;
    for (auto iter = L.begin(); iter != L.end(); ) {
      auto X = *iter;
      num = calculate_support(set_part_map[X]);
      if (!C[X]) {
        iter = L.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  inline void compute_partition_on_demand(long long X) {
    if (set_part_map.find(X) != set_part_map.end()) {
      return;
    }
    auto& source = parents[X];
    compute_partition_on_demand(source.first);
    compute_partition_on_demand(source.second);

    auto& lhs = set_part_map[source.first];
    auto& rhs = set_part_map[source.second];
    
    // set_part_map[X].reserve(15000);
    set_part_map[X].reserve(lhs.size()+rhs.size());
    if (lhs.size() < rhs.size()) {
      multiply_partitions(lhs, rhs, set_part_map[X]);
    } else {
      multiply_partitions(rhs, lhs, set_part_map[X]);
    }
    // cout<<"compute_partition_on"<<X<<endl;
  }




  void get_deps(std::set<std::pair<std::string, std::string>> &allFD){
      std::vector<std::vector<int>> tmp;
      tmp.reserve(FD.size());

      for (int i = 0; i < FD.size(); ++i) {
        auto& item =FD[i];
        auto lhs = decode_to_vector(item.first);
        auto rhs = item.second;
        lhs.push_back(rhs);
        tmp.emplace_back(std::move(lhs));
      }
      std::sort(tmp.begin(), tmp.end(), [](const std::vector<int> &lhs, const std::vector<int> &rhs) -> bool {
        auto iter1 = lhs.begin();
        auto iter2 = rhs.begin();
        while (iter1 != lhs.end() && iter2 != rhs.end())
        {
          if (*iter1 < *iter2)
            return true;
          if (*iter1 > *iter2)
            return false;
          ++iter1;
          ++iter2;
        }
        return (iter2 != rhs.end());
      });
      // cout<<attributes.size()<<endl;
      // for(int i = 0; i < attributes.size(); ++i){
      //     std::cout << attributes[i] << endl;
      //   }
      
      // std::vector<std::pair<std::string, std::string>> dependencies;
      
      for (int i = 0; i < tmp.size(); ++i) {
        auto& dep = tmp[i];
    
        // for(int i = 0; i < dep.size(); ++i){
        //   std::cout << dep[i] << attributes[dep[i]] << endl;
        // }
        std::string leftHandSide = "";

        for (int j = 0; j < dep.size() - 1; ++j) {
          // ofs << dep[j] + 1 << " ";
          // depStream << attributes[dep[j]] << " ";

          leftHandSide += attributes[dep[j]] + " ";
        }
        // ofs << "-> " << dep[dep.size() - 1] + 1<< "\n";
        // depStream << "-> " << attributes[dep[dep.size() - 1]] << "\n";
        
        allFD.insert(std::make_pair(leftHandSide, attributes[dep[dep.size() - 1]]));
        // std::cout << leftHandSide << " -> " << attributes[dep[dep.size() - 1]] << std::endl;
      }


  }

};

#endif  // TANE_H
