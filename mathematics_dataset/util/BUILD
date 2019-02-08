# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

py_library(
    name = "combinatorics",
    srcs = ["combinatorics.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
    ],
)

py_test(
    name = "combinatorics_test",
    srcs = ["combinatorics_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":combinatorics",
        # absl/testing:absltest dep,
    ],
)

py_library(
    name = "composition",
    srcs = ["composition.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":combinatorics",
        ":display",
        "//mathematics_dataset/sample:number",
        "//mathematics_dataset/sample:ops",
        "//mathematics_dataset/sample:polynomials",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_test(
    name = "composition_test",
    srcs = ["composition_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":composition",
        # absl/testing:absltest dep,
        # sympy dep,
    ],
)

py_library(
    name = "display",
    srcs = ["display.py"],
    srcs_version = "PY2AND3",
    deps = [
        # sympy dep,
    ],
)

py_test(
    name = "display_test",
    srcs = ["display_test.py"],
    deps = [
        ":display",
        # absl/testing:absltest dep,
        # sympy dep,
    ],
)

py_library(
    name = "probability",
    srcs = ["probability.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
        # sympy dep,
    ],
)

py_test(
    name = "probability_test",
    srcs = ["probability_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":probability",
        # absl/testing:absltest dep,
        # sympy dep,
    ],
)
