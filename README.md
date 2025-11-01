# strong-decomposition
A method to achieve the MLE decomposition of DAGs. 

This code provides a collection of functions for generating, manipulating, and analyzing Directed Acyclic Graphs (DAGs). It is particularly suited for working with DAGs. The toolkit supports functionalities such as inducing path detection, and checks for strong convex hull.

## Features

Implement advanced concepts including moralization, induced paths, and tests for t-removability and c-removability.
Provide methods for finding convex hulls (e.g., via minimal separators or absorbing inducing paths).


## Usage

1. **Random DAG Generation**: 
    ```python
    from your_script import Generate_DAG
    
    n = 8  # Number of nodes
    edge_density = 0.2  # Probability of an edge between any pair of nodes
    
    DAG = Generate_DAG(n, edge_density)
    ```
   
2. **Identify c-convex hull**:
    ```python
    from your_script import CMCSA111_new
    
    G = {
        'r1': {'m1': 'a', 'm2': 'b'},
        'r2': {'m1': 'c', 'm2': 'd', 'm3': 'e', 'r3': 'j'},
        'r3': {'m2': 'f'},
        'm1': {'m3': 'g'},
        'm2': {'m3': 'h'},
        'm3': {}
    }
    R = ['m1', 'm2', 'm3']
    
    c_convex = CMCSA111_new(G, R)
    print(c_convex)
    ```
3. **Achieve Strong Decomposition**:
    ```python
    from your_script import ICRSA
    
    G = {
        'r1': {'m1': 'a', 'm2': 'b'},
        'r2': {'m1': 'c', 'm2': 'd', 'm3': 'e', 'r3': 'j'},
        'r3': {'m2': 'f'},
        'm1': {'m3': 'g'},
        'm2': {'m3': 'h'},
        'm3': {}
    }
   
    H = EDC(G)
    print(H)
    ```
4. **Visualize Graph**:
    ```python
    from your_script import plot_result
    
    plot_result(DAG, 'output.png', title='Random DAG')
    ```

## Examples

Run the following examples in the main block:
```python
if __name__ == "__main__":
    G = {
        'r1': {'m1': 'a', 'm2': 'b'},
        'r2': {'m1': 'c', 'm2': 'd', 'm3': 'e', 'r3': 'j'},
        'r3': {'m2': 'f'},
        'm1': {'m3': 'g'},
        'm2': {'m3': 'h'},
        'm3': {}
    }
    R = ['m1', 'm2', 'm3']
    print(CMCSA111_new(G, R))
    print(EDC(G))
   
```

## Notes

- The code also includes several functions for more advanced graph manipulations, such as finding Markov blankets, inducing paths, and moralizing the DAG.
- The primary focus is on exploring inducing paths in DAGs and how it relates to the removability of certain vertex sets.

## License

This project is open-source and available under the MIT License.
