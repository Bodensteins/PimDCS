#ifndef PIMTORCH_DEVICE_H
#define PIMTORCH_DEVICE_H
#pragma once

#include <tuple>
#include <torch/torch.h>
#include <map>
#include "pim_array_config.h"
#include <torch/custom_class.h>

namespace PIM {

struct device
{
    // read out logic state of the array
    // arr is storing the conductance (float type)
    virtual at::Tensor read_states(at::Tensor arr) = 0; 


    // write data to arr. 
    //data is logic states and arr is float conductance
    virtual void write_states(at::Tensor &arr, at::Tensor data) = 0;


};

/**
 * 
 * This device can be multiple bit
 * but program by PV operation (program & verify)
 * which has higher reliability.
 * */
struct digitDevice: device
{
    const pim_array_pro_config *conf;
    digitDevice(const pim_array_pro_config *conf = &pro_decf()): conf(conf) {}

    
};

/**
 * analog device.
 * program without verifying.
 * similar to the neurosim .
 * */
struct analogDevice
{

};



}

#endif