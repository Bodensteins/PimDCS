//
// Created by 周恒 on 2020/12/22.
//

#include "../pim_linear.h"
#include "../pim_conv.h"
#include "../logic_array_interface.h"


template<typename PimType>
void declare_pim_linear_func(pybind11::module &m, const std::string &typestr) {
  using Class = PIM::PimLinearFunction<PimType>;
  std::string pyclass_name = typestr + std::string("PimLinearFunction");
  pybind11::class_<Class>(m, pyclass_name.c_str())
      .def("forward", &Class::forward)
      .def("backward", &Class::backward);
}

void declare_pim_linear(pybind11::module &m) {
  auto pim_linear = pybind11::class_<PIM::PimLinearImpl, std::shared_ptr<PIM::PimLinearImpl>>(m, "PimLinear");
  pim_linear.def(pybind11::init<int64_t, int64_t, int64_t, PIM::PimArrayType>())
      .def("reset", &PIM::PimLinearImpl::reset)
      .def("reset_parameters", &PIM::PimLinearImpl::reset_parameters)
      .def("pretty_print", &PIM::PimLinearImpl::pretty_print)
      .def("forward", &PIM::PimLinearImpl::forward)
      .def("sync_weight", &PIM::PimLinearImpl::sync_weight)
      .def("train", &PIM::PimLinearImpl::train)
      .def("print_pim_weight", &PIM::PimLinearImpl::print_pim_weight)
      .def_readwrite("weight", &PIM::PimLinearImpl::weight)
      .def_readwrite("bias", &PIM::PimLinearImpl::bias);
}


template<typename PimType>
void declare_pim_conv_func(pybind11::module &m, const std::string &typestr) {
  using Class = PIM::PimConv2dFunction<PimType>;
  std::string pyclass_name = typestr + std::string("PimConv2dFunction");
  pybind11::class_<Class>(m, pyclass_name.c_str())
      .def("forward", &Class::forward)
      .def("backward", &Class::backward);
}

void declare_pim_conv(pybind11::module &m) {
  auto pim_conv = pybind11::class_<PIM::PimConv2dImpl, std::shared_ptr<PIM::PimConv2dImpl>>(m, "PimConv2d");
  pim_conv.def(pybind11::init<std::vector<int64_t>, std::vector<int64_t>, int64_t, PIM::PimArrayType>())
      .def("reset", &PIM::PimConv2dImpl::reset)
      .def("reset_parameters", &PIM::PimConv2dImpl::reset_parameters)
      .def("pretty_print", &PIM::PimConv2dImpl::pretty_print)
      .def("forward", &PIM::PimConv2dImpl::forward)
      .def("sync_weight", &PIM::PimConv2dImpl::sync_weight)
      .def("train", &PIM::PimConv2dImpl::train)
      .def_readwrite("weight", &PIM::PimConv2dImpl::weight)
      .def_readwrite("bias", &PIM::PimConv2dImpl::bias);
}

TORCH_LIBRARY(pimtorch, m) {
  m.class_<PIM::SimpleLogicArray>("SimpleLogicArray")
      .def(torch::init<int64_t, int64_t>())
      .def("write_cell", &PIM::SimpleLogicArray::write_cell)
      .def("read_cell", &PIM::SimpleLogicArray::read_cell)
      .def("mm", &PIM::SimpleLogicArray::mm)
      .def("mv", &PIM::SimpleLogicArray::mv)
      .def("write_row", &PIM::SimpleLogicArray::write_row)
      .def("read_row", &PIM::SimpleLogicArray::read_row)
      .def("write_mat", &PIM::SimpleLogicArray::write_mat)
      .def("read_mat", &PIM::SimpleLogicArray::read_mat)
      .def("dot_column", &PIM::SimpleLogicArray::dot_column)
      .def("nmv", &PIM::SimpleLogicArray::nmv)
      .def("resize", &PIM::SimpleLogicArray::resize)
      .def("sizes", &PIM::SimpleLogicArray::sizes);
  m.class_<PIM::PimArrayPtr>("PimArrayPtr")
      .def(torch::init());
  m.class_<PIM::PimArrayPtrList>("PimArrayPtrList")
      .def(torch::init());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::enum_<PIM::PimArrayType>(m, "PimArrayType")
    .value("SimpleLogicArray", PIM::PimArrayType::simple_logic_array)
    .value("WbLogicArray", PIM::PimArrayType::wb_logic_array);

  pybind11::class_<PIM::SimpleLogicArray>(m, "SimpleLogicArray")
    .def(pybind11::init<int64_t, int64_t>())
    .def("write_cell", &PIM::SimpleLogicArray::write_cell)
    .def("read_cell", &PIM::SimpleLogicArray::read_cell)
    .def("mm", &PIM::SimpleLogicArray::mm)
    .def("mv", &PIM::SimpleLogicArray::mv)
    .def("write_row", &PIM::SimpleLogicArray::write_row)
    .def("read_row", &PIM::SimpleLogicArray::read_row)
    .def("write_mat", &PIM::SimpleLogicArray::write_mat)
    .def("read_mat", &PIM::SimpleLogicArray::read_mat)
    .def("dot_column", &PIM::SimpleLogicArray::dot_column)
    .def("nmv", &PIM::SimpleLogicArray::nmv)
    .def("resize", &PIM::SimpleLogicArray::resize)
    .def("sizes", &PIM::SimpleLogicArray::sizes);

  declare_pim_linear_func<PIM::SimpleLogicArray>(m, "Simple");
  declare_pim_linear(m);
  declare_pim_conv_func<PIM::SimpleLogicArray>(m, "Simple");
  declare_pim_conv(m);
}


