#include <pybind11/pybind11.h>

#include "pointpillars_main.h"

namespace py = pybind11;

PYBIND11_MODULE(pointpillars_interface, m) {
    py::class_<pointpillars_main>(m, "pointpillars_main")
        .def(py::init<std::string, std::string>())
        .def("doinit", &pointpillars_main::doinit)
	.def("doinfer", &pointpillars_main::doinfer);

}

