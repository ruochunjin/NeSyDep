#include "stringutil.h"
#include <algorithm>
#include <cctype>

// std::ptr_fun/std::not1 were removed in C++17 (libc++); use a lambda.
static bool not_space(unsigned char c) { return !std::isspace(c); }

// trim from start
std::string& ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    return s;
}

// trim from end
std::string& rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

// trim from both ends
std::string& trim(std::string &s) {
    return ltrim(rtrim(s));
}

std::string concat(int count, ...) {
    std::stringstream ss;
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
        ss << va_arg(args, const char*);
    }
    va_end(args);
    return ss.str();
}

std::string concatCsv(int count, ...) {
    std::stringstream ss;
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
        ss << va_arg(args, const char*);
        if (i < count - 1) {
            ss << ",";
        }
    }
    va_end(args);
    return ss.str();
}
