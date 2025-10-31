"""
quantum_ai_demo.py

Author: Amanuel Alemu Zewdu
Purpose: Demonstrate a simple quantum circuit using IBM Quantum Experience for AI optimization.
Requirements: qiskit
"""

from qiskit import QuantumCircuit, Aer, execute

# Create a 3-qubit quantum circuit
qc = QuantumCircuit(3,3)
qc.h(0)
qc.cx(0,1)
qc.cx(0,2)
qc.measure_all()

# Simulate
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator).result()
counts = result.get_counts(qc)
print("Quantum Circuit Output Counts:", counts)

# Potential AI application: faster combinatorial optimization (e.g., feature selection, route optimization)
