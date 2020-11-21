//
// Created by 周恒 on 2020/11/19.
//

#ifndef PIMTORCH_LOGIC_ARRAY_INTERFACE_H
#define PIMTORCH_LOGIC_ARRAY_INTERFACE_H

#pragma once

#include <torch/torch.h>
#include <vector>
#include <iostream>

/**
 * LogicArrayInterface is a API for libtorch to map operations to PIM simulator.
 */
class LogicArrayInterface {
public:
  /**
   * Constructs a logic array (2D Tensor).
   * @param rowSize row size
   * @param colSize column size
   * @param dtype data type
   */
  LogicArrayInterface(int64_t rowSize, int64_t colSize, torch::ScalarType dtype=torch::kFloat64) : rowSize(rowSize), colSize(colSize) {}

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
  virtual torch::Tensor read_row(int64_t row, int64_t col, int64_t size) = 0;          //从x行从y列开始读取size个列的数据

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
  virtual torch::Tensor nmv(int64_t row, int64_t col, const torch::Scalar &value, int64_t size) = 0; //数值乘以向量，需要输入操作的row号

  virtual torch::IntArrayRef sizes() const = 0;
  virtual torch::Tensor& resize(torch::IntArrayRef size) const = 0;
  virtual std::ostream& print(std::ostream &os) const = 0;

  friend std::ostream& operator << ( std::ostream &os, const LogicArrayInterface &arr);

protected:
  int64_t rowSize, colSize;  // matrix of size: m rows and n columns.
};

std::ostream& operator << ( std::ostream &os, const LogicArrayInterface &arr)
{
  return arr.print(os);
}

class SimpleLogicArray : public LogicArrayInterface {
public:
  SimpleLogicArray(int64_t rowSize, int64_t colSize, torch::ScalarType dtype = torch::kFloat64)
      : LogicArrayInterface(rowSize, colSize, dtype) {
    arr = torch::zeros({rowSize, colSize}, dtype);
  }

  void write_cell(int64_t row, int64_t col, const torch::Scalar &value) override {
    arr[row][col] = value;
  }

  torch::Scalar read_cell(int64_t row, int64_t col) override {
    return arr[row][col].item();
  }

  torch::Tensor mv(const torch::Tensor &vec) override {
    std::vector<torch::Tensor> l;
    for (int64_t i = 0; i < colSize; ++i) {
      l.push_back(dot_column(vec, i));
    }
    return torch::stack(l, 0);
  }

  void write_row(int64_t row, int64_t col, const torch::Tensor &vec) override {
    int64_t len = vec.sizes()[0];
    assert(len > 0 && row < rowSize && col + len - 1 < colSize);
    arr.index_put_({torch::full(len, row, torch::kLong), torch::arange(col, col+len, torch::kLong)}, vec);
  }

  torch::Tensor read_row(int64_t row, int64_t col, int64_t size) override {
    assert(size > 0 && row < rowSize && col + size - 1 < colSize);
    return arr.index({row}).slice(0, col, col + size);
  }

  torch::Tensor dot_column(const torch::Tensor &vec, int64_t col) override {
    return arr.select(1, col).dot(vec);
  }

  torch::Tensor nmv(int64_t row, int64_t col, const torch::Scalar &value, int64_t size) override {
    return arr.index({row}).mul(value).slice(0, col, col + size);
  }

  torch::Tensor& resize(torch::IntArrayRef size) const override {
    return arr.resize_(size);
  }

  torch::IntArrayRef sizes() const override {
    return arr.sizes();
  }

  std::ostream& print(std::ostream &os) const override {
    return os << arr;
  }

  ~SimpleLogicArray() {}

  friend std::ostream& operator << ( std::ostream &os, const SimpleLogicArray &arr);

private:
  torch::Tensor arr;
};

std::ostream& operator << ( std::ostream &os, const SimpleLogicArray &arr) {
  return arr.print(os);
}


//void ComputeError(float_type *correct, float_type* logic_out, int64_t len) {
//  for (int64_t i = 0; i < len; i++) {
//    if (abs(correct[i] - logic_out[i]) > 1e-6) {
//      printf("[%d] correct vs. logic = %f vs %f\n", i, correct[i], logic_out[i]);
//    }
//  }
//}
#endif //PIMTORCH_LOGIC_ARRAY_INTERFACE_H
