#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


import numpy as np
from typing import Tuple
from ucimlrepo import fetch_ucirepo


from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[2]:


# Define a dataclass to hold the important hyperparameters
@dataclass(frozen=True)
class ModelHyperparameters:
    n_estimators: int = field(default=100, metadata={"min": 10, "max": 100})
    max_depth: int = field(default=1, metadata={"min": 1, "max": 50})
    min_samples_split: int = field(default=2, metadata={"min": 2, "max": 20})
    min_samples_leaf: int = field(default=1, metadata={"min": 1, "max": 20})

    def __hash__(self) -> int:
        return hash(
            (
                self.n_estimators,
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
            )
        )

    def validate(self) -> None:
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            metadata = field_def.metadata
            if "min" in metadata and value < metadata["min"]:
                raise ValueError(f"{field_name} should be at least {metadata['min']}")
            if "max" in metadata and value > metadata["max"]:
                raise ValueError(f"{field_name} should be at most {metadata['max']}")


# Define a function to create a ModelHyperparameters object with random values within the ranges
def create_random_hyperparameters() -> ModelHyperparameters:
    return ModelHyperparameters(
        n_estimators=random.randint(10, 1000),
        max_depth=random.randint(1, 50),
        min_samples_split=random.randint(2, 20),
        min_samples_leaf=random.randint(1, 20),
    )


# In[3]:


# Define function for loading and splitting the data
def load_and_split_data_iris(
    test_size: float = 0.5, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def load_and_split_data_wine(
    test_size: float = 0.5, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Fetch the dataset
    wine_quality = fetch_ucirepo(id=186)

    # Extract features and targets
    X = wine_quality.data.features.to_numpy()
    y = wine_quality.data.targets.to_numpy().ravel()

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def load_and_split_data_wine_rebalanced(
    test_size: float = 0.5, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Fetch the dataset
    wine_quality = fetch_ucirepo(id=186)

    # Extract features and targets
    X = wine_quality.data.features.to_numpy()
    y = wine_quality.data.targets.to_numpy().ravel()

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Apply SMOTE to rebalance the training data
    smote = SMOTE(
        sampling_strategy="auto",
        k_neighbors=min(5, len(np.unique(y_train)) - 1),
        random_state=random_state,
    )
    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    except ValueError as e:
        # Fallback to reduce k_neighbors if not enough samples are available
        smote = SMOTE(
            sampling_strategy="auto", k_neighbors=1, random_state=random_state
        )
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test


X_train, X_test, y_train, y_test = load_and_split_data_wine_rebalanced()


# Define function to train the RandomForestClassifier
def train_random_forest(
    params: ModelHyperparameters, random_state: int = 42
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        min_samples_leaf=params.min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


# Define function to evaluate the model
def evaluate_model(clf: RandomForestClassifier) -> float:
    y_pred = clf.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    return f1_macro


@lru_cache(maxsize=1064)
def get_fitness(params: ModelHyperparameters) -> float:
    model = train_random_forest(params, 42)
    fitness = evaluate_model(model)
    return fitness


# In[ ]:


# Mutation function for hyperparameters
def mutate(param: ModelHyperparameters, mut_prob: float = 0.05) -> ModelHyperparameters:
    def mutate_param(field_name: str, val: int, metadata: dict) -> int:
        if random.random() < mut_prob:
            min_val = metadata.get("min", val)
            max_val = metadata.get("max", val)
            return random.randint(min_val, max_val)
        return val

    return ModelHyperparameters(
        n_estimators=mutate_param(
            "n_estimators",
            param.n_estimators,
            ModelHyperparameters.__dataclass_fields__["n_estimators"].metadata,
        ),
        max_depth=(
            mutate_param(
                "max_depth",
                param.max_depth if param.max_depth is not None else 1,
                ModelHyperparameters.__dataclass_fields__["max_depth"].metadata,
            )
            if param.max_depth is not None
            else None
        ),
        min_samples_split=mutate_param(
            "min_samples_split",
            param.min_samples_split,
            ModelHyperparameters.__dataclass_fields__["min_samples_split"].metadata,
        ),
        min_samples_leaf=mutate_param(
            "min_samples_leaf",
            param.min_samples_leaf,
            ModelHyperparameters.__dataclass_fields__["min_samples_leaf"].metadata,
        ),
    )


# Crossover function for two hyperparameter sets
def crossover(
    params1: ModelHyperparameters, params2: ModelHyperparameters, cross_prob: float
) -> ModelHyperparameters:

    if random.random() > cross_prob:
        return random.choice([params1, params2])

    return ModelHyperparameters(
        n_estimators=random.choice([params1.n_estimators, params2.n_estimators]),
        max_depth=random.choice([params1.max_depth, params2.max_depth]),
        min_samples_split=random.choice(
            [params1.min_samples_split, params2.min_samples_split]
        ),
        min_samples_leaf=random.choice(
            [params1.min_samples_leaf, params2.min_samples_leaf]
        ),
    )


# In[5]:


def search(
    maxGen: int = 100,
    popSize: int = 10,
    tournSize: int = 2,
    mutProb: float = 0.05,
    crossProb: float = 0.8,
):

    highscore = -1.0
    fittest = (None, -1)
    gen = 0

    pop = [create_random_hyperparameters() for _ in range(popSize)]

    while gen < maxGen:

        pop.sort(key=lambda x: get_fitness(x), reverse=True)
        bestSol = pop[0]
        bestFit = get_fitness(bestSol)

        if bestFit > fittest[1]:
            fittest = bestSol, bestFit

        print(f"Generation {gen} - Best Solution: {pop[0]}")
        print(f"Fitness: {get_fitness(pop[0])}")
        breadingPop = []

        while len(breadingPop) < popSize / 2:

            tournPool = random.choices(pop, k=tournSize)
            tournPool.sort(key=lambda x: get_fitness(x), reverse=True)
            breadingPop.append(tournPool[0])

        pop = []

        while len(pop) < popSize:

            mom, dad = random.choices(breadingPop, k=2)
            child = mutate(crossover(mom, dad, crossProb), mutProb)
            pop.append(child)

        gen += 1

    return fittest


# In[6]:


search()

