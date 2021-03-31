#ifndef PIMTORCH_LOGIC_ARRAY_INTERFACE_H
#define PIMTORCH_LOGIC_ARRAY_INTERFACE_H

#pragma once

#include <torch/torch.h>
#include <torch/custom_class.h>
#include <iostream>

namespace PIM {

  enum class PimArrayType {
    simple_logic_array, pim_array, only_counters_pim_array, pim_array_pro
  };

 std::ostream &operator << ( std::ostream& stream, const PimArrayType &type)
 {
   const std::string name[] = {
       "PIM::simple_logic_array",
       "PIM::pim_array",
       "PIM::only_counters_pim_array"
   };
   return stream << name[(int)type];
 }

/**
 * LogicArrayInterface is a API for libtorch to map operations to PIM simulator.
 */
  class LogicArrayInterface : public torch::CustomClassHolder {
  public:
    /**
     * Constructs a logic array (2D Tensor).
     * @param rowSize row size
     * @param colSize column size
     * @param dtype data type
     */
    LogicArrayInterface(int64_t rowSize, int64_t colSize, const torch::TensorOptions& options = {})
        : rowSize(rowSize), colSize(colSize) {}

    /**
     * Copy constructor.
     * @param other
     */
    LogicArrayInterface(const LogicArrayInterface& other)
        : rowSize(other.rowSize), colSize(other.colSize) {}

    /**
     * Move constructor.
     * @param other
     */
    LogicArrayInterface(LogicArrayInterface&& other) {
      rowSize = std::exchange(other.rowSize, 0);
      colSize = std::exchange(other.colSize, 0);
    }
    /**
     * Writes a cell.
     * @param row index of row
     * @param col index of column
     * @param value value to write
     */
    virtual void write_cell(int64_t row, int64_t col, const torch::Scalar &value) = 0;

    /**
     * Reads a cell value.
     * @param row index of row
     * @param col index of column
     * @return torch::Scalar
     */
    virtual torch::Scalar read_cell(int64_t row, int64_t col) = 0;

    /**
     * Writes a vector vec at specific position.
     * Note that, deep copy should be used.
     * @param row index of row
     * @param col index of column
     * @param vec vector to write
     */
    virtual void write_row(int64_t row, int64_t col, const torch::Tensor &vec) = 0;

    /**
     * Reads a row values with a specific size.
     * @param row index of row
     * @param col index of column
     * @param size read size
     * @return torch::Tensor
     */
    virtual torch::Tensor read_row(int64_t row, int64_t col, int64_t size) = 0;

    /**
     * Writes a matrix(2D Tensor).
     * Note that, deep copy should be used.
     * @param mat matrix to write
     */
    virtual void write_mat(const torch::Tensor &mat) = 0;

    /**
     * Reads a matrix(2D Tensor)
     * @return torch::Tensor
     */
    virtual torch::Tensor read_mat() = 0;


    /**
     * Performs a matrix multiplication of the input matrix mat and self matrix.
     * @param mat input matrix
     * @return torch::Tensor
     */
    virtual torch::Tensor mm(const torch::Tensor &mat) = 0;

    /**
     * Performs a matrix-vector product of self matrix and the vector vec.
     * @param vec
     * @return torch::Tensor
     */
    virtual torch::Tensor mv(const torch::Tensor &vec) = 0;

    /**
     * Performs a dot product of input vec and specific column.
     * @param vec input vector
     * @param col index of column
     * @return torch::Tensor
     */
    virtual torch::Tensor dot_column(const torch::Tensor &vec, int64_t col) = 0;

    /**
     * Multiplies each element of the selected row with the scalar value and return a
     * new resulting tensor which start from selected column and with a length of size.
     * @param row index of row
     * @param col index of column
     * @param size size of resulting tensor
     * @return torch::Tensor
     */
    virtual torch::Tensor nmv(int64_t row, int64_t col, const torch::Scalar &value, int64_t size) = 0;

    virtual torch::IntArrayRef sizes() const = 0;

    virtual torch::Tensor &resize(torch::IntArrayRef size) const = 0;

    virtual std::ostream &print(std::ostream &os) const = 0;

    friend std::ostream &operator<<(std::ostream &os, const LogicArrayInterface &arr);


  protected:
    int64_t rowSize, colSize;  // matrix of size: m rows and n columns.
  };

//  std::ostream &operator<<(std::ostream &os, const LogicArrayInterface &arr) {
//    return arr.print(os);
//  }


  class SimpleLogicArray : public LogicArrayInterface {
  public:
    SimpleLogicArray(int64_t rowSize, int64_t colSize, const torch::TensorOptions& options = {})
        : LogicArrayInterface(rowSize, colSize, options) {
      arr = torch::zeros({rowSize, colSize}, options);
//      std::cout << "call normal constructor. (" << this << ")" << std::endl;
    }

    SimpleLogicArray(const SimpleLogicArray &other) : LogicArrayInterface(other) {
      arr = other.arr.clone();
//      std::cout << "call copy constructor. (" << this << " <- "<< &other <<")" << std::endl;
    }

    SimpleLogicArray(SimpleLogicArray &&other) : LogicArrayInterface(other) {
      arr = std::move(other.arr);
//      std::cout << "call move constructor. (" << this << " <- "<< &other <<")" << std::endl;
    }

    void write_cell(int64_t row, int64_t col, const torch::Scalar &value) override {
      arr[row][col] = value;
    }

    torch::Scalar read_cell(int64_t row, int64_t col) override {
      return arr[row][col].item();
    }

    torch::Tensor mm(const torch::Tensor &mat) override {
//    std::vector<torch::Tensor> l;
//    for (int64_t i = 0; i < mat.size(0); ++i) {
//      l.push_back(mv(mat.select(0, i)));
//    }
//    return torch::stack(l, 0);
      return torch::mm(mat, arr);
    }

    torch::Tensor mv(const torch::Tensor &vec) override {
//    std::vector<torch::Tensor> l;
//    for (int64_t i = 0; i < colSize; ++i) {
//      l.push_back(dot_column(vec, i));
//    }
//    return torch::stack(l, 0);
      return torch::mv(arr.t(), vec);
    }

    void write_row(int64_t row, int64_t col, const torch::Tensor &vec) override {
      int64_t len = vec.sizes()[0];
      assert(len > 0 && row < rowSize && col + len - 1 < colSize);
      arr.index_put_({torch::full(len, row, torch::kLong), torch::arange(col, col + len, torch::kLong)}, vec);
    }

    torch::Tensor read_row(int64_t row, int64_t col, int64_t size) override {
      assert(size > 0 && row < rowSize && col + size - 1 < colSize);
      return arr.index({row}).slice(0, col, col + size);
    }

    void write_mat(const torch::Tensor &mat) override {
//    assert(torch::is_same_size(arr, mat));
    arr = mat.clone();
//      arr = mat;
    }

    torch::Tensor read_mat() override {
//    return arr.slice(0, row_begin, row_end);
      return arr;
    }

    torch::Tensor dot_column(const torch::Tensor &vec, int64_t col) override {
      return arr.select(1, col).dot(vec);
    }

    torch::Tensor nmv(int64_t row, int64_t col, const torch::Scalar &value, int64_t size) override {
      return arr.index({row}).mul(value).slice(0, col, col + size);
    }

    torch::Tensor &resize(torch::IntArrayRef size) const override {
      return arr.resize_(size);
    }

    torch::IntArrayRef sizes() const override {
      return arr.sizes();
    }

    std::ostream &print(std::ostream &os) const override {
      return os << arr;
    }

    SimpleLogicArray& operator=(const SimpleLogicArray& other) {
      if (this != &other) {
        rowSize = other.rowSize;
        colSize = other.colSize;
        arr = other.arr.clone();
      }
      return *this;
    }

    SimpleLogicArray& operator=(SimpleLogicArray&& other) {
      if (this != &other) {
        rowSize = std::exchange(other.rowSize, 0);
        colSize = std::exchange(other.colSize, 0);
        arr = std::move(other.arr);
      }
      return *this;
    }

    friend std::ostream &operator<<(std::ostream &os, const SimpleLogicArray &arr);

    ~SimpleLogicArray() override = default;
  private:
    torch::Tensor arr;
  };

 TORCH_LIBRARY(SimpleLogicArray, m) {
   m.class_<SimpleLogicArray>("SimpleLogicArray")
       .def(torch::init<int64_t, int64_t, torch::ScalarType>())
       .def("write_cell", &SimpleLogicArray::write_cell)
       .def("read_cell", &SimpleLogicArray::read_cell)
       .def("mm", &SimpleLogicArray::mm)
       .def("mv", &SimpleLogicArray::mv)
       .def("write_row", &SimpleLogicArray::write_row)
       .def("read_row", &SimpleLogicArray::read_row)
       .def("write_mat", &SimpleLogicArray::write_mat)
       .def("read_mat", &SimpleLogicArray::read_mat)
       .def("dot_column", &SimpleLogicArray::dot_column)
       .def("nmv", &SimpleLogicArray::nmv)
       .def("resize", &SimpleLogicArray::resize)
       .def("sizes", &SimpleLogicArray::sizes);
 }

 std::ostream &operator<<(std::ostream &os, const SimpleLogicArray &arr) {
   return arr.print(os);
 }


 typedef std::shared_ptr<LogicArrayInterface> PimPtr;

 class PimArrayPtr : public torch::CustomClassHolder {
   public:
     PimArrayPtr() = default;
     PimPtr ptr = nullptr;
 };
 TORCH_LIBRARY(PimArrayPtr, m) {
   m.class_<PimArrayPtr>("PimArrayPtr")
       .def(torch::init());
 }

 class PimArrayPtrList : public torch::CustomClassHolder {
   public:
     PimArrayPtrList() = default;
//      explicit PimArrayPtrList(int64_t size) : ptrs(std::vector<PimPtr>(size)) {}
     std::vector<PimPtr> ptrs;
 };

 TORCH_LIBRARY(PimArrayPtrList, m) {
   m.class_<PimArrayPtrList>("PimArrayPtrList")
       .def(torch::init());
 }
}
#endif //PIMTORCH_LOGIC_ARRAY_INTERFACE_H
