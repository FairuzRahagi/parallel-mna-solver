#include <iostream>
#include "circuit.hpp"
#include "mna.hpp"

// ---------------------------------------------------------------
// Circuit 1: Voltage divider
//   V1 = 5 V  (gnd=node0 → node1)
//   R1 = 1000 Ω  (node1 – node2)
//   R2 = 2000 Ω  (node2 – gnd)
//
//   Internal nodes: 1, 2  →  node_offset = 2
//   Voltage sources: 1    →  extra row at index 2
//   MNA size: 3×3
// ---------------------------------------------------------------
static void test_voltage_divider() {
    Circuit c;
    c.addVoltageSource(1, 0, 5.0);   // V1: node0(gnd) → node1, 5 V
    c.addResistor(1, 2, 1000.0);     // R1: node1 – node2
    c.addResistor(2, 0, 2000.0);     // R2: node2 – gnd

    MNA mna;
    mna.build(c);

    std::cout << "=== Circuit 1: Voltage Divider ===\n";
    std::cout << "MNA size: " << mna.getSize() << "x" << mna.getSize()
              << "  (expected 3x3)\n\n";
    mna.print();
    std::cout << "\n";
}

// ---------------------------------------------------------------
// Circuit 2: Simple series loop
//   V1 = 10 V  (gnd=node0 → node1)
//   R1 = 5 Ω   (node1 – gnd)
//
//   Internal nodes: 1  →  node_offset = 1
//   Voltage sources: 1 →  extra row at index 1
//   MNA size: 2×2
// ---------------------------------------------------------------
static void test_simple_loop() {
    Circuit c;
    c.addVoltageSource(1, 0, 10.0);  // V1: gnd → node1, 10 V
    c.addResistor(1, 0, 5.0);        // R1: node1 – gnd

    MNA mna;
    mna.build(c);

    std::cout << "=== Circuit 2: Simple Loop ===\n";
    std::cout << "MNA size: " << mna.getSize() << "x" << mna.getSize()
              << "  (expected 2x2)\n\n";
    mna.print();
    std::cout << "\n";
}

int main() {
    test_voltage_divider();
    test_simple_loop();
    return 0;
}
