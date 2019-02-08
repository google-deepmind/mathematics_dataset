# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

py_library(
    name = "algebra",
    srcs = ["algebra.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//mathematics_dataset:example",
        "//mathematics_dataset/sample:linear_system",
        "//mathematics_dataset/sample:number",
        "//mathematics_dataset/sample:ops",
        "//mathematics_dataset/sample:polynomials",
        "//mathematics_dataset/util:composition",
        "//mathematics_dataset/util:display",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_test(
    name = "algebra_test",
    srcs = ["algebra_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":algebra",
        # absl/testing:absltest dep,
        "//mathematics_dataset/sample:polynomials",
        # six dep,
        # sympy dep,
    ],
)

py_library(
    name = "arithmetic",
    srcs = ["arithmetic.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//mathematics_dataset:example",
        "//mathematics_dataset/sample:arithmetic",
        "//mathematics_dataset/sample:number",
        "//mathematics_dataset/sample:ops",
        "//mathematics_dataset/util:composition",
        "//mathematics_dataset/util:display",
        # sympy dep,
    ],
)

py_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":arithmetic",
        # absl/testing:absltest dep,
        # sympy dep,
    ],
)

py_library(
    name = "calculus",
    srcs = ["calculus.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//mathematics_dataset:example",
        "//mathematics_dataset/sample:polynomials",
        "//mathematics_dataset/util:composition",
        "//mathematics_dataset/util:display",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_test(
    name = "calculus_test",
    srcs = ["calculus_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calculus",
        # tensorflow dep,
    ],
)

py_library(
    name = "comparison",
    srcs = ["comparison.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//mathematics_dataset:example",
        "//mathematics_dataset/sample:number",
        "//mathematics_dataset/sample:ops",
        "//mathematics_dataset/util:composition",
        "//mathematics_dataset/util:display",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_library(
    name = "measurement",
    srcs = ["measurement.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":train_test_split",
        "//mathematics_dataset:example",
        "//mathematics_dataset/sample:number",
        "//mathematics_dataset/util:composition",
        "//mathematics_dataset/util:display",
        # six dep,
        # sympy dep,
    ],
)

py_library(
    name = "modules",
    srcs = ["modules.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":algebra",
        ":arithmetic",
        ":calculus",
        ":comparison",
        ":measurement",
        ":numbers",
        ":polynomials",
        ":probability",
    ],
)

py_library(
    name = "numbers",
    srcs = ["numbers.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//mathematics_dataset:example",
        "//mathematics_dataset/sample:number",
        "//mathematics_dataset/util:composition",
        "//mathematics_dataset/util:display",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_library(
    name = "polynomials",
    srcs = ["polynomials.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//mathematics_dataset:example",
        "//mathematics_dataset/sample:number",
        "//mathematics_dataset/sample:ops",
        "//mathematics_dataset/sample:polynomials",
        "//mathematics_dataset/util:composition",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_library(
    name = "probability",
    srcs = ["probability.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":train_test_split",
        "//mathematics_dataset:example",
        "//mathematics_dataset/util:combinatorics",
        "//mathematics_dataset/util:composition",
        "//mathematics_dataset/util:display",
        "//mathematics_dataset/util:probability",
        # numpy dep,
        # six dep,
    ],
)

py_library(
    name = "train_test_split",
    srcs = ["train_test_split.py"],
    srcs_version = "PY2AND3",
    deps = [],
)
