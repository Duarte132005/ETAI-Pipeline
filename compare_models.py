from src.evaluate import corrected_paired_test

adaboost = [0.674, 0.661, 0.686, 0.685, 0.691]
oblique_random_forest = [0.682, 0.658, 0.696, 0.690, 0.684]
result = corrected_paired_test(
    adaboost,
    oblique_random_forest,
    test_train_ratio=1 / 4,
)

print(result)