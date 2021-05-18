#ifndef PIMTORCH_PRUE_ARRAY_INTERFACE_H
#define PIMTORCH_PRUE_ARRAY_INTERFACE_H
#pragma once

#include "logic_array_interface.h"
#include "pim_array_example.h"

namespace PIM
{
    /**
   * Creates a PIM array which contains a 2D Tensor.
   * @param arr shared_ptr of LogicArrayInterface
   * @param arr_size shape of 2D Tensor
   * @param pim_type type of PIM array
   * @param options Tensor options
   */
    void create_pim_array(PimArrayPtr &arr_ptr, torch::ExpandingArray<2> arr_size, PimArrayType pim_type,
                          const torch::TensorOptions &options = {})
    {
        switch (pim_type)
        {
            case PimArrayType::simple_logic_array:
                arr_ptr.ptr = std::make_shared<SimpleLogicArray>((*arr_size)[0], (*arr_size)[1], options);
                break;
            case PimArrayType::pim_array_pro:
                arr_ptr.ptr = std::make_shared<pimArrayPro>((*arr_size)[0], (*arr_size)[1], options);
                break;
            default:
                TORCH_INTERNAL_ASSERT(false, "create array, pimArrayType not support!");
                break;
        }
    }

    /**
   * Creates a list of PIM array which represents a 3D Tensor.
   * @param arr_list reference of PimArrayList
   * @param arr_shape shape of 3D Tensor
   * @param pim_type type of PIM array
   * @param options Tensor options
   */
    void create_pim_array_list(PimArrayPtrList &array_ptrs, torch::ExpandingArray<3> arr_shape, PimArrayType pim_type,
                               const torch::TensorOptions &options = {})
    {
        switch (pim_type)
        {
            case PimArrayType::simple_logic_array:
                array_ptrs.ptrs.resize((*arr_shape)[0]);
                at::parallel_for(0, (*arr_shape)[0], 0, [&](int64_t start, int64_t end) {
                    for (int64_t i = start; i < end; i++)
                    {
                        array_ptrs.ptrs[i] = std::make_shared<SimpleLogicArray>((*arr_shape)[1], (*arr_shape)[2], options);
                    }
                });
                break;
            case PimArrayType::pim_array_pro:
                array_ptrs.ptrs.resize((*arr_shape)[0]);
                at::parallel_for(0, (*arr_shape)[0], 0, [&](int64_t start, int64_t end) {
                    for (int64_t i = start; i < end; i++)
                    {
                        array_ptrs.ptrs[i] = std::make_shared<pimArrayPro>((*arr_shape)[1], (*arr_shape)[2], options);
                    }
                });
                break;
            default:
                TORCH_INTERNAL_ASSERT(false, "create arraylist, pimArrayType not support!");
                break;
        }
    }

} // namespace PIM
#endif