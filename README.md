# cQAOA
The purpose of this GitHub is to begin exploring the notion of cyclic QAOA (**cQAOA**).  cQAOA is inspired by https://arxiv.org/pdf/2403.01034 which draws on notions of the adiabatic theorem https://en.wikipedia.org/wiki/Adiabatic_theorem , quantum annealing https://en.wikipedia.org/wiki/Quantum_annealing , and many body localization https://en.wikipedia.org/wiki/Many-body_localization .

## The Algorithm
The structure of the cQAOA algorithm is as follows:
-Select a low energy reference state.  (For initialization, you can choose any random state or begin from a single round of QAOA)

-Form the circuit of layers $\gamma$ - $\beta$ - $\alpha(t)$ where $\alpha(t)$ is the reference layer at time step t.  This is the trotterization of the Hamiltonian $H(t)=H_{cost}+H_{ref}(t)$ with boundary conditions $H_{ref}(0) = H_{ref}$ and $H_{ref}(T) = 0$.  Parameters of Trotterization should be chosen for the competing goals of 1. Maintaining adiabaticity 2. Efficiently generating a solution.  A simple initial approach is to linearly decrease the strength of the $\alpha$ layer linearly until it vanishes.  In this problem, the $\alpha$ layer is simply implemented by adjusting the local field parameters $h_i$

-Perform measurements to determine lowest energy state(s).

-Replace the low energy reference state and repeat.


## Completed:
QAOA Implementation following Cirq example
Visualization of H(0) for cQAOA for a reference state following a round of QAOA
Rough implementation of cQAOA using gradient descent in analogy to Cirq example
Some inital testing

## Questions:

-How best to involve classical optimization? (My approach is very, very rough)

-How to select strength of $\alpha$ layer, referred to as refBias in cQAOA implementation notebook?

-How to best balance number of layers between minimal circuit depth and sufficient adiabaticity?

-Is cQAOA better than QAOA? What metrics would show this?  Is it possible to analytically calculate an approximation ratio?

## To do list:
### 1. Reimplement in more generality/ abstraction and with code optimizations. 
IE def QAOA(costHamiltonian(costParameters),qaoaParameters) and def cQAOA(costHamiltonian(costParameters),cqaoaParameters) for general costHamiltonians, costParameters, qaoaParameters, and cqaoaParameters

### 2. Move towards systematic testing:
Optimizing qaoaParameters, cqaoaParameters for general costParameters

Define these parameters precisely and determine how they relate to one another

### 3. Consider larger/ different problems with more local minima.
IE def toyModel(costParameters)

### 4. Implement To do list items 1 and 3 in helper python functions that can be called within Jupyter Notebooks for readability.
