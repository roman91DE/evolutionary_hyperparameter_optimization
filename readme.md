# Genetic Hyperparameter Optimization for RandomForest Classifier

## Overview
This project demonstrates a prototype implementation for optimizing hyperparameters of machine learning models using a Genetic Algorithm. Specifically, it is used to optimize the hyperparameters of a RandomForestClassifier for multiclass classification tasks, with datasets such as Iris and Wine Quality. The script leverages `imbalanced-learn` to handle dataset imbalances and employs a genetic approach to evolve the hyperparameters to achieve optimal performance.

## Features
- Utilizes Genetic Algorithm for hyperparameter optimization, including mutation and crossover operations.
- Implements caching to speed up repeated evaluations of hyperparameter combinations.
- Handles imbalanced datasets using SMOTE to resample the data.
- Includes support for multiple datasets, including the Iris dataset and the Wine Quality dataset from the UCI repository.

## Requirements
- Python 3.8+
- Required packages:
  - `scikit-learn`
  - `imbalanced-learn`
  - `ucimlrepo`

Install the requirements with:

```sh
pip install scikit-learn imbalanced-learn ucimlrepo
```

## Usage
Run the script to start the genetic algorithm optimization:

```sh
python hyperparameter_optimizer.py
```

The main optimization loop can be controlled using the following parameters:
- `maxGen`: Maximum number of generations (default is 100)
- `popSize`: Population size per generation (default is 10)
- `tournSize`: Tournament size for selection (default is 2)
- `mutProb`: Mutation probability (default is 0.05)
- `crossProb`: Crossover probability (default is 0.8)

The script will display the best hyperparameter combination found during the evolution and its fitness score.

## Genetic Algorithm Details
- **Population Initialization**: A population of random hyperparameter sets is initialized.
- **Selection**: Tournament selection is used to select individuals for breeding.
- **Crossover and Mutation**: Crossover generates offspring by combining two parents, and mutation introduces variations to the hyperparameters to maintain diversity.
- **Fitness Evaluation**: Each individual's fitness is determined by the F1 macro score on the test data.

## Example Output
```
Generation 0 - Best Solution: ModelHyperparameters(n_estimators=100, max_depth=10, ...)
Fitness: 0.85
...
Generation 99 - Best Solution: ModelHyperparameters(n_estimators=150, max_depth=20, ...)
Fitness: 0.92
```

## License
This project is licensed under the MIT License.

