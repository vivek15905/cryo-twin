# 🫀 In Silico Cryo-Twin: Physics-Informed Neural Networks for Organ Preservation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Status](https://img.shields.io/badge/Status-Prototype-green)
![Domain](https://img.shields.io/badge/Domain-AI4Science-purple)

### **The "Why": Solving the Organ Shortage with AI**
Cryopreservation (freezing organs) is the "Holy Grail" of transplant medicine, but it fails because of **thermal stress**: if an organ cools unevenly, it cracks. Traditional simulations (FEA/CFD) take hours to model this too slow for real-time control.

**This project builds a differentiable "Digital Twin" that learns the laws of thermodynamics.**
Instead of using training data, this model uses a **Physics-Informed Neural Network (PINN)** to "reason" about heat transfer by minimizing the residual of the Heat Equation PDE. It allows for millisecond-level simulation of tissue temperature states, enabling real-time optimization of cooling protocols.

[🚀 **View the Executable Notebook Here**](./Digital_twin_prototype.ipynb)

---

##  Scientific Results

### 1. The "Reasoning" Engine (Heat Diffusion)
The model successfully learned the **1D Heat Equation** ($\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$) without seeing a single data point. It deduced the temperature distribution purely from the physics loss function.

![Digital Twin Heatmap](heatmap.png)
*Figure 1: Spatiotemporal heatmap of tissue cooling. The model captures the diffusion of heat from the center (red) to the boundaries (blue) over time.*

### 2. Validation Against Physics
To verify the model isn't hallucinating, we check specific time snapshots. The temperature profiles (below) perfectly match the analytical Gaussian decay expected in thermodynamic systems.

![Validation Snapshots](snapshots.png)
*Figure 2: Temperature profiles at t=0.0, 0.2, 0.5, and 0.8s. The smooth decay proves the network has learned the diffusive nature of the PDE.*

### 3. Diagnostic: Where does Physics Break?
Transparency is key in AI for Science. The **Residual Map** below shows the absolute error of the differential equation at every point in space-time.

![Residual Map](residual_map.png)
*Figure 3: Physics Residual Map. Dark areas indicate near-perfect adherence to conservation laws. Bright spots highlight high-gradient regions (initial conditions) where adaptive sampling is needed.*

---

## Technical Approach

### The "Unsupervised" Loss Function
We do not train on a dataset. We train on the **laws of physics**.
The loss function is a sum of two terms:
![Loss Function](https://latex.codecogs.com/svg.latex?\bg_white&space;\Large&space;\mathcal{L}=\mathcal{L}_{PDE}+\mathcal{L}_{IC})
Where $\mathcal{L}_{PDE}$ forces the network to obey the Heat Equation:
```python
# The "Reasoning" Step
u_t = grad(u, t)      # Time derivative
u_xx = grad(u_x, x)   # Spatial derivative
residual = u_t - alpha * u_xx
loss = mean(residual**2)

```
### System Architecture
The following diagram illustrates how the Physics Loss acts as a "teacher," correcting the network via backpropagation without needing ground-truth labels.

```mermaidgraph LR
    A["Input (x, t)"] --> B("Neural Network<br/>PINN")
    B --> C["Predicted Temp (u)"]
    
    C --> D{"Automatic Differentiation<br/>(Autograd)"}
    
    D -->|"du/dt"| E["Time Derivative"]
    D -->|"d²u/dx²"| F["Spatial Derivative"]
    
    E & F --> G["Physics Residual<br/>(Heat Equation)"]
    G --> H(("Loss Function"))
    
    C -->|"Boundary Data"| H
    H -->|"Backprop"| B
    
    style A fill:#8e44ad,stroke:#fff,stroke-width:2px,color:#fff
    style G fill:#2c3e50,stroke:#fff,stroke-width:2px,color:#fff
    style H fill:#d35400,stroke:#fff,stroke-width:4px,color:#fff

```
👤 About the Author
Vivek Pendem, Mechanical Engineer & Researcher | Focus: AI for Science & Bioprinting Working at the intersection of high-performance computing, thermodynamics, and biological preservation.
