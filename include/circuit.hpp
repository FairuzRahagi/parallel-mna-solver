#ifndef MNA_PROJECT_CIRCUIT_HPP
#define MNA_PROJECT_CIRCUIT_HPP
#include <vector>

struct Resistor {
    int node_pos, node_neg;
    double resistance;
};

struct VoltageSource {
    int node_pos, node_neg;
    double voltage;
    int branch_idx;
};

struct CurrentSource {
    int node_pos, node_neg;
    double current;
};

class Circuit {
public:
    int num_nodes = 0;

    std::vector<Resistor>      resistors;
    std::vector<VoltageSource> voltage_sources;
    std::vector<CurrentSource> current_sources;

    void addResistor(int node_pos, int node_neg, double resistance) {
        resistors.push_back({node_pos, node_neg, resistance});
        num_nodes = std::max(num_nodes, std::max(node_pos, node_neg) + 1);
    }

    void addVoltageSource(int node_pos, int node_neg, double voltage) {
        int k = static_cast<int>(voltage_sources.size());
        voltage_sources.push_back({node_pos, node_neg, voltage, k});
        num_nodes = std::max(num_nodes, std::max(node_pos, node_neg) + 1);
    }

    void addCurrentSource(int node_pos, int node_neg, double current) {
        current_sources.push_back({node_pos, node_neg, current});
        num_nodes = std::max(num_nodes, std::max(node_pos, node_neg) + 1);
    }

    // MNA matrix dimension: (num_nodes - 1) + num_voltage_sources
    int getSize() const {
        return (num_nodes - 1) + static_cast<int>(voltage_sources.size());
    }
};

#endif // MNA_PROJECT_CIRCUIT_HPP
