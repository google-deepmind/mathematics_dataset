# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

py_library(
    name = "arithmetic",
    srcs = ["arithmetic.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":number",
        ":ops",
        "//mathematics_dataset/util:combinatorics",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":arithmetic",
        ":number",
        ":ops",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # six dep,
        # sympy dep,
    ],
)

py_library(
    name = "linear_system",
    srcs = ["linear_system.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":number",
        ":ops",
        ":polynomials",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_test(
    name = "linear_system_test",
    srcs = ["linear_system_test.py"],
    flaky = 1,  # TODO(b/116779004): Fix test and remove this tag.
    deps = [
        ":linear_system",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # six dep,
        # sympy dep,
    ],
)

py_library(
    name = "number",
    srcs = ["number.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//mathematics_dataset/util:display",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_test(
    name = "number_test",
    srcs = ["number_test.py"],
    deps = [
        ":number",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # six dep,
        # sympy dep,
    ],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":number",
        # absl/logging dep,
        "//mathematics_dataset/util:display",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_test(
    name = "ops_test",
    srcs = ["ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
        # absl/testing:absltest dep,
        # six dep,
        # sympy dep,
    ],
)

py_library(
    name = "polynomials",
    srcs = ["polynomials.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":number",
        ":ops",
        "//mathematics_dataset/util:combinatorics",
        # numpy dep,
        # six dep,
        # sympy dep,
    ],
)

py_test(
    name = "polynomials_test",
    srcs = ["polynomials_test.py"],
    deps = [
        ":polynomials",
        # absl/testing:parameterized dep,
        # numpy dep,
        # six dep,
        # sympy dep,
        # tensorflow dep,
    ],
)
