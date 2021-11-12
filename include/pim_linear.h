#ifndef PIMTORCH_PIM_LINEAR_H
#define PIMTORCH_PIM_LINEAR_H

#pragma once
#include <torch/torch.h>
#include "pim_utils.h"

using namespace torch;
using namespace torch::autograd;
using namespace torch::nn;

namespace PIM {
    template<typename PimType>
    class PimLinearFunction : public Function<PimLinearFunction<PimType>> {

    public:
        // Note that both forward and backward are static functions

        /**
         * Forward function for linear module.
         *
         * @param ctx autograd context
         * @param wb pim array for weight and bias
         * @param wb_t pim array for transposed weight
         * @param prev pim array to save input data
         *             or buffer array to save input data  (two design choices)
         * @param input input tensor
         * @param weight weight tensor
         * @param bias bias tensor(optional)
         * @param is_training training status
         * @param train_mode If you need training, set to PIMRunMode::train, then prev array will be written.
         *                  if set to PIMRunMode::inference, these arrays will not be written.
         * @return
         */
        static Tensor forward(
            AutogradContext *ctx, PimArrayPtr &wb, PimArrayPtr &wb_t, PimArrayPtr &prev,
            const Tensor &input, const Tensor &weight, const c10::optional<Tensor> &bias,
            bool is_training, PIMRunMode train_mode = PIMRunMode::train)
        {
            ctx->save_for_backward({input, weight, bias.has_value() ? bias.value() : Tensor()});

            ctx->saved_data["wb_t_ptr"] = c10::make_intrusive<PimArrayPtr>(wb_t);
            ctx->saved_data["prev_ptr"] = c10::make_intrusive<PimArrayPtr>(prev);
            ctx->saved_data["train_mode"] = int(train_mode);

            // write parameters to PIM if is trainable
            if (is_training && train_mode != PIMRunMode::inference && weight.requires_grad())
            {
                prev.ptr->write_mat(input);
            }

            Tensor pim_input = input;
            if (bias.has_value())
            {
                ConstantPad2d m(ConstantPad2dOptions({0, 1, 0, 0}, 1));
                pim_input = m(input);
            }
            Tensor pim_output = wb.ptr->mm(pim_input.detach()); // shape of wb_ptr: (in_features, out_features)

            return pim_output;
        }

        /**
         * Backward function for linear module.
         *
         *  Tensor grad_input = grad_output.mm(weight);
         *  Tensor grad_weight = grad_output.t().mm(input);
         *
         * @param ctx autograd context
         * @param grad_outputs tensor list for grad outputs
         * @return tensor list of grad outputs
         */
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
        {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto bias = saved[2];
            PIMRunMode train_mode = static_cast<PIMRunMode>(ctx->saved_data["train_mode"].toInt());

            Tensor grad_output = grad_outputs[0];
            Tensor grad_bias = Tensor();
            if (bias.defined())
            {
                grad_bias = grad_output.sum(0);
            }

            if (train_mode == PIMRunMode::fast_mode_train)
            {
                Tensor grad_input = grad_output.mm(weight);
                Tensor grad_weight = grad_output.t().mm(input);
                return {Tensor(), Tensor(), Tensor(), grad_input, grad_weight, grad_bias, Tensor(), Tensor()};
            }
            else
            {
                // shape of wb_t_ptr: (out_features, in_features)
                Tensor pim_grad_input = ctx->saved_data["wb_t_ptr"].toCustomClass<PimArrayPtr>()->ptr->mm(grad_output.detach());
                Tensor pim_grad_weight = ctx->saved_data["prev_ptr"].toCustomClass<PimArrayPtr>()->ptr->mm(grad_output.t().detach());
                return {Tensor(), Tensor(), Tensor(), pim_grad_input, pim_grad_weight, grad_bias, Tensor(), Tensor()};
            }
        }
    };


    class TORCH_API PimLinearImpl : public Cloneable<PimLinearImpl> {
    public:
        PimLinearImpl(int64_t in_features, int64_t out_features, int64_t batch_size, PimArrayType pim_type,
                      PIMRunMode train_mode, const TensorOptions op = {})
            : PimLinearImpl(batch_size, pim_type, LinearOptions(in_features, out_features), train_mode, op) {}

        explicit PimLinearImpl(int64_t batch_size, PimArrayType pim_type,
                               const LinearOptions &options_, PIMRunMode train_mode, const TensorOptions op = {})
            : options(options_), batch_size(batch_size), pim_type(pim_type), train_mode(train_mode)
        {
            reset();
            this->to(op.device());
            create_pim_array(wb_ptr, {options_.bias() ? options_.in_features() + 1 : options_.in_features(), options_.out_features()},
                             pim_type, op, std::to_string(options_.bias() ? options_.in_features() + 1 : options_.in_features()) + "x" + std::to_string(options_.out_features()) + "_(wb)_");
            if (train_mode != PIMRunMode::inference)
            {
                create_pim_array(wb_t_ptr, {options_.out_features(), options_.in_features()}, pim_type,
                                 op, std::to_string(options_.out_features()) + "x" + std::to_string(options_.in_features()) + "_(wb_t)_");
                
                if (train_mode == PIMRunMode::train_transientInBuffer)
                    create_pim_array(prev_ptr, {batch_size, options_.in_features()}, PimArrayType::simple_logic_array,
                                 op, std::to_string(batch_size) + "x" + std::to_string(options_.in_features()) + "_(prev_ptr)_");
                else
                    create_pim_array(prev_ptr, {batch_size, options_.in_features()}, pim_type,
                                 op, std::to_string(batch_size) + "x" + std::to_string(options_.in_features()) + "_(prev_ptr)_");
            }
            sync_weight();
        }

        void reset() override
        {
            weight = register_parameter("weight", torch::empty({options.out_features(), options.in_features()}));
            if (options.bias())
            {
                bias = register_parameter("bias", torch::empty(options.out_features()));
            }
            else
            {
                bias = register_parameter("bias", {}, /*requires_grad=*/false);
            }
            reset_parameters();
        }

        void reset_parameters()
        {
            torch::nn::init::kaiming_uniform_(weight, std::sqrt(5)); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
            if (bias.defined())
            {
                int64_t fan_in, fan_out;
                std::tie(fan_in, fan_out) =
                    torch::nn::init::_calculate_fan_in_and_fan_out(weight);
                const auto bound = 1 / std::sqrt(fan_in);
                torch::nn::init::uniform_(bias, -bound, bound);
            }
        }

        /// Pretty prints the `Linear` module into the given `stream`.
        void pretty_print(std::ostream &stream) const override
        {
            stream << std::boolalpha
                   << "PIM::PimLinear(in_features=" << options.in_features()
                   << ", out_features=" << options.out_features()
                   << ", bias=" << options.bias() << ")";
            if (print_detail_)
            {
                stream << "array" << std::endl;
                wb_ptr.ptr->print(stream);
                stream << "array_t" << std::endl;
                if (wb_t_ptr.ptr != nullptr)
                    wb_t_ptr.ptr->print(stream);
                stream << "prev" << std::endl;
                prev_ptr.ptr->print(stream);
            }
        }

        /// Transforms the `input` tensor by multiplying with the `weight` and
        /// optionally adding the `bias`, if `with_bias` is true in the options.
        Tensor forward(const Tensor &input)
        {
            switch (pim_type)
            {
            case PimArrayType::simple_logic_array:
                return PimLinearFunction<SimpleLogicArray>::apply(wb_ptr, wb_t_ptr, prev_ptr, input, weight,
                                                                  options.bias() ? bias : c10::optional<Tensor>(), is_training(), train_mode);
            case PimArrayType::pim_array_pro:
                return PimLinearFunction<pimArrayPro>::apply(wb_ptr, wb_t_ptr, prev_ptr, input, weight,
                                                             options.bias() ? bias : c10::optional<Tensor>(), is_training(), train_mode);
            case PimArrayType::pim_array_fast:
                return PimLinearFunction<pimArrayFast>::apply(wb_ptr, wb_t_ptr, prev_ptr, input, weight,
                                                              options.bias() ? bias : c10::optional<Tensor>(), is_training(), train_mode);
            default:
                TORCH_INTERNAL_ASSERT(false, "pimlinear, forward type not support!")
            }
        }

        /**
         * sync_weight from tensor weight to our pim array.
         * 
         * */
        void sync_weight()
        {
            //if define weight sync with pim. then the weight will be re-sync with pim array after pim array is quantized sync with weight.
            if (pro_decf().weights_sync_with_pim)
            {
                if (bias.defined())
                {
                    wb_ptr.ptr->write_mat(torch::cat({weight.t(), bias.unsqueeze(0)}, 0)); // write transposed weight
                    torch::Tensor w_idx = torch::arange(weight.size(1), TensorOptions(torch::kLong).device(weight.device()));
                    weight.data() = wb_ptr.ptr->read_mat().index_select(0, w_idx).t();
                    bias.data() = wb_ptr.ptr->read_row(weight.size(1), 0, weight.size(0));
                }
                else
                {
                    wb_ptr.ptr->write_mat(weight.t());
                    weight.data() = wb_ptr.ptr->read_mat().t();
                }
                if (wb_t_ptr.ptr != nullptr)
                    wb_t_ptr.ptr->write_mat(weight);
            }
            else
            {
                double max_weights = pro_decf().max_weight_value;

                if (bias.defined())
                {
                    if (pro_decf().weights_tensor_trunc)
                    {
                        weight.data() = limit_weight_with_max(weight, max_weights);
                        bias.data() = limit_weight_with_max(bias, max_weights);
                    }
                    wb_ptr.ptr->write_mat(torch::cat({weight.t(), bias.unsqueeze(0)}, 0)); // write transposed weight
                }
                else
                {
                    if (pro_decf().weights_tensor_trunc)
                    {
                        weight.data() = limit_weight_with_max(weight, max_weights);
                    }
                    wb_ptr.ptr->write_mat(weight.t());
                }
                if (wb_t_ptr.ptr != nullptr)
                    wb_t_ptr.ptr->write_mat(weight);
            }
        }

        void check_weight_sync()
        {
            if (!torch::allclose(weight, wb_t_ptr.ptr->read_mat(), 1e-05, 1e-06))
            {
                TORCH_INTERNAL_ASSERT(false, "wb_t_ptr not correct");
            }
        }

        //    void train(bool on = true) override {
        //      if (!on) {
        //        sync_weight();
        //      }
        //      is_training_ = on;
        //    }

        void print_detail(bool on = true)
        {
            print_detail_ = on;
        }

        /// The options used to configure this module.
        LinearOptions options;

        /// The learned weight. float 
        Tensor weight;

        Tensor bias;

        bool print_detail_{false};

        PIMRunMode train_mode;

        PimArrayPtr wb_ptr;
        PimArrayPtr wb_t_ptr;
        PimArrayPtr prev_ptr;
        PimArrayType pim_type;
        int64_t batch_size;
    };

    /// A `ModuleHolder` subclass for `LinearImpl`.
    /// See the documentation for `LinearImpl` class to learn what methods it
    /// provides, and examples of how to use `Linear` with `torch::nn::LinearOptions`.
    /// See the documentation for `ModuleHolder` to learn about PyTorch's
    /// module storage semantics.
    TORCH_MODULE(PimLinear);
}
#endif //PIMTORCH_PIM_LINEAR_H
