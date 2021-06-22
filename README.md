# Simple Mesh Example

Domain decomposition for a simple mesh, using Trilinos' Tpetra and Zoltan2. 

![Mesh](ExampleMesh.png)

- [x] Exodus File I/O
    - [x] Read Exodus File into Memory
    - [x] Partition Elements and Nodes
        - [x] METIS
        - [ ] ParMETIS
    - [x] Write Domain Decomposed Exodus File
        - [x] Element Partitions as Element Blocks
        - [ ] Node Partitions as NodeSets
- [ ] Partial Differential Equations (PDE) and Laplace's Equation