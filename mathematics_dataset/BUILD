package(default_visibility = ["//visibility:public"])

# Placeholder for internal Python version compatibility macro.

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

py_library(
    name = "example",
    srcs = ["example.py"],
    srcs_version = "PY2AND3",
    deps = ["//mathematics_dataset/util:composition"],
)

py_library(
    name = "generate_lib",
    srcs = ["generate.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        "//mathematics_dataset/modules",
        # six dep,
    ],
)

py_binary(
    name = "generate",
    srcs = ["generate.py"],
    deps = [":generate_lib"],
)

py_binary(
    name = "generate_to_file",
    srcs = ["generate_to_file.py"],
    deps = [
        ":generate_lib",
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        # six dep,
    ],
)

py_test(
    name = "generate_test",
    srcs = ["generate_test.py"],
    deps = [
        ":generate_lib",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # six dep,
    ],
)
