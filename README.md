# KNN Imputer With Hamming Distance
Filling missed categorical data-points with the most common value among nearest neighbors using KNN-based imputation and Hamming as a distance metric.

## Example Usage

```
from knn_imputer_with_hamming import KnnImputerWithHamming

knn_imputer = KnnImputerWithHamming(data)
knn_imputer.calculate_hamming_distance()
knn_imputer.impute_data(n=30, threshold=0.8)

print(knn_imputer.features)
```
*A quick solution for small datasets. But if you are working with large datasets, you are welcome to contribute and optimize the code.