#ifndef MNA_PROJECT_MNA_HPP
#define MNA_PROJECT_MNA_HPP
#include <vector>
#include "circuit.hpp"

using Matrix = std::vector<std::vector<double>>;
using Vec    = std::vector<double>;

class MNA {
public:
    void build(const Circuit& c);

    const Matrix& getA() const { return A; }
    const Vec&    getB() const { return b; }
    int           getSize() const { return static_cast<int>(b.size()); }
    void          print() const;

private:
    Matrix A;
    Vec    b;
};

#endif // MNA_PROJECT_MNA_HPP
