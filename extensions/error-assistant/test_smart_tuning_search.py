
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression

from ml_platform_error_assistant.smart_tuning_search import (
    get_smart_search_space,
    build_smart_trials,
    describe_changes,
)


def main():
    knn = KNeighborsClassifier()

    basic = get_smart_search_space(
        knn, level="basic"
    )
    advanced = get_smart_search_space(
        knn, level="advanced"
    )

    assert "n_neighbors" in basic
    assert "weights" in basic
    assert "p" not in basic
    assert "p" in advanced

    basic_trials = build_smart_trials(
        knn, level="basic"
    )
    advanced_trials = build_smart_trials(
        knn, level="advanced"
    )

    assert basic_trials[0] == {}
    assert advanced_trials[0] == {}
    assert len(basic_trials) <= 8
    assert len(advanced_trials) <= 20

    print("KNN BASIC AND ADVANCED SEARCH PASSED")

    forest = RandomForestRegressor()

    forest_space = get_smart_search_space(
        forest, level="advanced"
    )

    assert "max_depth" in forest_space
    assert "min_samples_leaf" in forest_space

    print("RANDOM FOREST SEARCH PASSED")

    ridge = Ridge()

    assert "alpha" in get_smart_search_space(ridge)

    print("RIDGE SEARCH PASSED")

    linear = LinearRegression()

    assert build_smart_trials(linear) == [{}]

    print("LINEAR REGRESSION DEFAULT HANDLING PASSED")

    changes = describe_changes(
        knn, {"n_neighbors": 10, "weights": "distance"}
    )

    assert len(changes) == 2
    assert changes[0]["recommended"] == 10

    print("CHANGED-ONLY PARAMETER DISPLAY PASSED")
    print("ALL SMART TUNING SEARCH TESTS PASSED")


if __name__ == "__main__":
    main()
