#pragma once

#include <pybind11/pybind11.h>
#include <boost/multiprecision/cpp_int.hpp>

namespace py = pybind11;

namespace pybind11 { namespace detail {

template <>
struct type_caster<boost::multiprecision::cpp_int> {
public:
    PYBIND11_TYPE_CASTER(boost::multiprecision::cpp_int, _("cpp_int"));

    // Python -> C++
    bool load(handle src, bool) {
        if (!py::isinstance<py::int_>(src))
            return false;
        value = boost::multiprecision::cpp_int(py::reinterpret_borrow<py::int_>(src).cast<std::string>());
        return true;
    }

    // C++ -> Python
    static handle cast(const boost::multiprecision::cpp_int &src,
                       return_value_policy /* policy */,
                       handle /* parent */) {
        return py::int_(py::str(src.str())).release();
    }
};

}} // namespace pybind11::detail
