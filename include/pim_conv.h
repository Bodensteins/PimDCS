#ifndef PIMTORCH_PIM_CONV_H
#define PIMTORCH_PIM_CONV_H

#pragma once

#include <torch/torch.h>
#include "pim_utils.h"

using namespace torch;
using namespace torch::autograd;
using namespace torch::nn;
using namespace at::native;
namespace F = torch::nn::functional;
static int parallel_for_size = 0;

namespace PIM {
    template<typename T>
    static inline T div_rtn(T x, T y) {
        int q = x / y;
        int r = x % y;
        if ((r != 0) && ((r < 0) != (y < 0))) --q;
        return q;
    }

    Tensor remove_padding(const Tensor &input, IntArrayRef padding) {
        IntArrayRef out_size = {input.size(0),
                                input.size(1),
                                input.size(2) - padding[2] - padding[3],
                                input.size(3) - padding[0] - padding[1]};
        Tensor mask = F::pad(torch::ones(out_size).toType(torch::kBool), F::PadFuncOptions(padding.vec()));
        return torch::masked_select(input, mask).view(out_size);
    }

    Tensor insert_zeros(const Tensor &input, IntArrayRef stride) {
        if (stride[0] == 1 && stride[1] == 1) {
            return input;
        } else {
            auto w = input.new_zeros(stride);
            w[0][0] = 1;
            Tensor output = F::conv_transpose2d(
                    input, w.expand({input.size(1), 1, stride[0], stride[1]}),
                    F::ConvTranspose2dFuncOptions().stride(stride).groups(input.size(1)));
            return remove_padding(output, {0, stride[0] - 1, 0, stride[1] - 1});
        }
    }

    static inline void conv2d_shape_check(
            const Tensor &input,
            const Tensor &weight,
            const Tensor &bias,
            IntArrayRef kernel_size,
            IntArrayRef stride,
            IntArrayRef padding) {
        const int64_t kernel_height = kernel_size[0];
        const int64_t kernel_width = kernel_size[1];
        const int64_t pad_height = padding[0];
        const int64_t pad_width = padding[1];
        const int64_t stride_height = stride[0];
        const int64_t stride_width = stride[1];
        TORCH_CHECK(
                kernel_width > 0 && kernel_height > 0,
                "kernel size should be greater than zero, but got kernel_height: ",
                kernel_height,
                " kernel_width: ",
                kernel_width);
        TORCH_CHECK(
                stride_width > 0 && stride_height > 0,
                "stride should be greater than zero, but got stride_height: ",
                stride_height,
                " stride_width: ",
                stride_width);
        if (weight.defined()) {
            TORCH_CHECK(
                    weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 4),
                    "non-empty 2D or 4D weight tensor expected, but got: ",
                    weight.sizes());
            if (bias.defined()) {
                check_dim_size(bias, 1, 0, weight.size(0));
            }
        } else {
            TORCH_CHECK(false, "weight tensor is undefined");
        }
        const int64_t ndim = input.dim();
        const int64_t dim_batch = 0;
        const int64_t dim_planes = 1;
        const int64_t dim_height = 2;
        const int64_t dim_width = 3;

        // Allow for empty batch size but not other dimensions
        bool valid_empty = ndim == 4 && input.size(dim_batch) == 0 &&
                input.size(dim_planes) != 0 && input.size(dim_height) != 0 &&
                input.size(dim_width) != 0;

        TORCH_CHECK(
                (input.numel() > 0 || valid_empty) && ndim == 4,
                "non-empty 4D input tensor expected but got: ",
                input.sizes());

        const int64_t input_height = input.size(dim_height);
        const int64_t input_width = input.size(dim_width);

        const int64_t exact_input_height = input_height + 2 * pad_height;
        const int64_t exact_input_width = input_width + 2 * pad_width;

        TORCH_CHECK(
                exact_input_height >= kernel_height && exact_input_width >= kernel_width,
                "Calculated padded input size per channel: (",
                exact_input_height,
                " x ",
                exact_input_width,
                "). ",
                "Kernel size: (",
                kernel_height,
                " x ",
                kernel_width,
                "). Kernel size can't be greater than actual input size");

        const int64_t output_height =
                div_rtn<int64_t>(exact_input_height - kernel_height, stride_height) + 1;
        const int64_t output_width =
                div_rtn<int64_t>(exact_input_width - kernel_width, stride_width) + 1;

        TORCH_CHECK(
                output_width >= 1 && output_height >= 1,
                "Given input size per channel: (",
                input_height,
                " x ",
                input_width,
                "). "
                "Calculated output size per channel: (",
                output_height,
                " x ",
                output_width,
                "). Output size is too small");

        if (weight.defined()) {
            int64_t n_input_plane = weight.size(1);
            if (weight.dim() == 2) {
                n_input_plane /= (kernel_height * kernel_width);
            }
            check_dim_size(input, ndim, dim_planes, n_input_plane);
        }
    }


    template<typename PimType>
    class PimConv2dFunction : public Function<PimConv2dFunction<PimType>> {
            public:
                static torch::Tensor forward(
                        AutogradContext *ctx, PimArrayPtr &wb, PimArrayPtr &wb_t, PimArrayPtrList &prevs,
                        const Tensor &input, const Tensor &weight, const c10::optional<Tensor> &bias,
                        IntArrayRef stride, IntArrayRef padding, bool is_training, bool fast_mode = false) {

                    IntArrayRef kernel_size = weight.sizes().slice(2);
                    conv2d_shape_check(input, weight, bias.has_value() ? bias.value() : Tensor(), kernel_size, stride, padding);
                    const int64_t kernel_height = kernel_size[0];
                    const int64_t kernel_width = kernel_size[1];
                    const int64_t pad_height = padding[0];
                    const int64_t pad_width = padding[1];
                    const int64_t stride_height = stride[0];
                    const int64_t stride_width = stride[1];
                    const int64_t dim_planes = 1;
                    const int64_t dim_height = 2;
                    const int64_t dim_width = 3;
                    const int64_t n_input_plane = input.size(dim_planes);
                    const int64_t input_height = input.size(dim_height);
                    const int64_t input_width = input.size(dim_width);
                    const int64_t n_output_plane = weight.size(0);
                    const int64_t output_height =
                            (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
                    const int64_t output_width =
                            (input_width + 2 * pad_width - kernel_width) / stride_width + 1;
                    const int64_t batch_size = input.size(0);

                    ctx->save_for_backward({input, weight, bias.has_value() ? bias.value() : Tensor()});
                    ctx->saved_data["stride"] = std::vector<int64_t>(stride.vec());
                    ctx->saved_data["padding"] = std::vector<int64_t>(padding.vec());
                    ctx->saved_data["wb_t_ptr"] = c10::make_intrusive<PimArrayPtr>(wb_t);
                    ctx->saved_data["prev_ptrs"] = c10::make_intrusive<PimArrayPtrList>(prevs);
                    ctx->saved_data["fast_mode"] = fast_mode;

                    // write parameters to PIM if is trainable
                    if (is_training && weight.requires_grad()) {
                        for (int64_t i = 0; i < input.size(0); i++) {
                            prevs.ptrs[i]->write_mat(input[i].flip({1, 2}).reshape({input[i].size(0), -1}).t());
                        }

                        // update parameters, note that weight shape is (Cin * H * W, Cout)
                        //        if (bias.has_value()) {
                        //          wb.ptr->write_mat(torch::cat({
                        //                                           weight.permute({1, 2, 3, 0}).reshape({-1, n_output_plane}),
                        //                                           bias.value().unsqueeze(0)}, 0));
                        //        } else {
                        //          wb.ptr->write_mat(weight.permute({1, 2, 3, 0}).reshape({-1, n_output_plane}));
                        //        }
                        //        // flip kernel
                        //        wb_t.ptr->write_mat(weight.flip({2, 3}).permute({1, 0, 2, 3}).reshape({n_input_plane, -1}).t());
                    }

                    // gold result
                    //      Tensor output;
                    //      if (bias.has_value()) {
                    //        output = functional::conv2d(
                    //            input, weight, F::Conv2dFuncOptions().bias(bias.value()).stride(stride).padding(padding));
                    //      } else {
                    //        output = functional::conv2d(
                    //            input, weight, F::Conv2dFuncOptions().stride(stride).padding(padding));
                    //      }

                    auto pim_input = F::unfold(input,
                                               UnfoldOptions(kernel_size).padding(padding).stride(stride)).transpose(1, 2);
                    //      auto pim_output = torch::zeros({batch_size, pim_input.size(1), n_output_plane}, weight.options());

                    if (bias.has_value()) {
                        ConstantPad1d add_bias(ConstantPad1dOptions({0, 1}, 1));
                        pim_input = add_bias(pim_input);
                    }
                    //      at::parallel_for(0, batch_size, parallel_for_size, [&](int64_t start, int64_t end) {
                    //        for (int64_t i = start; i < end; i++) {
                    //          pim_output[i] = wb.ptr->mm(pim_input[i].detach());   // shape of wb_ptr: (in_features, out_features)
                    //        }
                    //      });
                    auto pim_output = wb.ptr->mm(pim_input.detach());
                    pim_output.transpose_(1, 2);
                    pim_output = F::fold(pim_output, FoldOptions({output_width, output_height}, {1, 1}));

                    //      if (!torch::allclose(output, pim_output, 1e-05, 1e-08)) {
                    //        TORCH_INTERNAL_ASSERT(false, "calculation error");
                    //      }
                    //      std::cout << "forward output" << std::endl;
                    //      Tensor err = output - pim_output;
                    //      std::cout << err << std::endl;

                    return pim_output;
                }

                static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
                    auto saved = ctx->get_saved_variables();
                    auto input = saved[0];
                    auto weight = saved[1];
                    auto bias = saved[2];
                    bool fast_mode = ctx->saved_data["fast_mode"].toBool();

                    const int64_t batch_size = input.size(0);
                    std::vector<int64_t> input_size({input.size(2), input.size(3)});
                    std::vector<int64_t> kernel_size({weight.size(2), weight.size(3)});
                    std::vector<int64_t> unfold_padding({kernel_size[0] - 1, kernel_size[1] - 1});
                    auto stride = ctx->saved_data["stride"].toIntVector();
                    auto padding = ctx->saved_data["padding"].toIntVector();

                    c10::intrusive_ptr<PimArrayPtr> wb_t_ptr = ctx->saved_data["wb_t_ptr"].toCustomClass<PimArrayPtr>();
                    c10::intrusive_ptr<PimArrayPtrList> prev_ptrs = ctx->saved_data["prev_ptrs"].toCustomClass<PimArrayPtrList>();

                    Tensor grad_output = grad_outputs[0];
                    Tensor grad_bias = Tensor();
                    if (bias.defined()) {
                        grad_bias = grad_output.sum({0, 2, 3});
                    }

                    if (fast_mode) {
                        int64_t in_ex = input.size(0);
                        int64_t in_ch = input.size(1);
                        int64_t out_ch = grad_output.size(1);
                        at::Tensor dLdZ_col = grad_output.transpose(0,3).reshape({out_ch, -1}); //合并transpose
                        at::Tensor X_unfold = F::unfold(input,
                                                        F::UnfoldFuncOptions(kernel_size).padding(padding).stride(stride))
                                                                .permute({1, 2, 0})
                                                                .reshape({in_ch * kernel_size[0] * kernel_size[1], -1})
                                                                .transpose(0, 1);
                        at::Tensor dW = dLdZ_col.matmul(X_unfold)
                                .reshape({out_ch, in_ch, kernel_size[0], kernel_size[1]});

                        // dX
                        at::Tensor W_col = weight.reshape({out_ch, -1}).transpose(0, 1);
                        at::Tensor dX_col = W_col.matmul(dLdZ_col)
                                .reshape({in_ch * kernel_size[0] * kernel_size[1], -1, in_ex})
                                .permute({2, 0, 1});
                        at::Tensor dX = F::fold(dX_col, F::FoldFuncOptions(input_size, kernel_size)
                        .padding(padding).stride(stride));
                        return {Tensor(), Tensor(), Tensor(), dX, dW, grad_bias,
                                Tensor(), Tensor(), Tensor(), Tensor()}; // number of returns should be equal to forward's args.
                    } else {
                        at::Tensor insert_dLdZ = insert_zeros(grad_output.permute({0, 1, 3, 2}), stride);

                        // grad of input
                        at::Tensor unf_dLdZ = F::unfold(insert_dLdZ,F::UnfoldFuncOptions(kernel_size)
                        .padding(unfold_padding)).transpose(1, 2);
                        at::Tensor dZ_ = wb_t_ptr->ptr->mm(unf_dLdZ).transpose(1, 2);
                        at::Tensor pim_grad_input = F::fold(dZ_, F::FoldFuncOptions(input_size, {1, 1}).padding(padding));
                        // grad of weight
                        at::Tensor swap_flip_X = input.flip({2, 3}).permute({1, 0, 2, 3}).transpose(1, 2);
                        at::Tensor unf_swap_dLdZ = F::unfold(insert_dLdZ.permute({1, 0, 2, 3}),
                                                             F::UnfoldFuncOptions(input_size)
                                                             .padding(unfold_padding)).transpose(1, 2);
                        std::vector<at::Tensor> dW_(batch_size);
                        auto dLdZ_chunks = unf_swap_dLdZ.chunk(batch_size, 2);

                        at::parallel_for(0, batch_size, parallel_for_size, [&](int64_t start, int64_t end) {
                            for (int64_t i = start; i < end; i++) {
                                dW_[i] = prev_ptrs->ptrs[i]->mm(dLdZ_chunks[i]).transpose(1, 2);
                            }
                        });
                        at::Tensor dW = torch::stack(dW_, 0).sum(0);
                        at::Tensor pim_grad_weight = F::fold(dW,F::FoldFuncOptions(kernel_size, {1, 1})
                        .padding(padding));

                        return {Tensor(), Tensor(), Tensor(), pim_grad_input, pim_grad_weight, grad_bias,
                                Tensor(), Tensor(), Tensor(), Tensor()}; // number of returns should be equal to forward's args.
                    }
                }
            };


    class TORCH_API PimConv2dImpl : public Cloneable<PimConv2dImpl> {
    public:
        PimConv2dImpl(ExpandingArray<4> input_shape,
                      ExpandingArray<2> kernel_size,
                      int64_t output_channels,
                      PimArrayType pim_type,
                      bool fast_mode = false,
                      const torch::TensorOptions &op = {})
                      : PimConv2dImpl(input_shape, pim_type,
                                      Conv2dOptions((*input_shape)[1], output_channels, kernel_size), fast_mode, op) {}

                                      explicit PimConv2dImpl(ExpandingArray<4> input_shape, PimArrayType pim_type,
                                                             const Conv2dOptions &options_, bool fast_mode = false, const torch::TensorOptions &op = {})
                                                             : input_shape(input_shape), pim_type(pim_type), options(options_), fast_mode(fast_mode){

            ExpandingArray<2> kernel_size = options_.kernel_size();
            const int64_t n_input_plane = options_.in_channels();
            const int64_t n_output_plane = options_.out_channels();

            reset();
            this->to(op.device());
            create_pim_array(wb_ptr, {
                options_.bias() ? (*kernel_size)[0] * (*kernel_size)[1] * n_input_plane + 1 :
                (*kernel_size)[0] * (*kernel_size)[1] * n_input_plane, n_output_plane
                }, pim_type, weight.options());
            create_pim_array(wb_t_ptr, {
                (*kernel_size)[0] * (*kernel_size)[1] * n_output_plane, n_input_plane
                }, pim_type, weight.options());
            create_pim_array_list(prev_ptrs, {
                (*input_shape)[0], (*input_shape)[2] * (*input_shape)[3], (*input_shape)[1]
                }, pim_type, weight.options());
            std::cout << weight << std::endl;
            sync_weight();
        }


        void reset() override {
            weight = register_parameter("weight",
                                        torch::empty({options.out_channels(),
                                                      options.in_channels(),
                                                      (*options.kernel_size())[0],
                                                      (*options.kernel_size())[1]}));
            if (options.bias()) {
                bias = register_parameter("bias", torch::empty(options.out_channels()));
            } else {
                bias = register_parameter("bias", {}, /*requires_grad=*/false);
            }
            reset_parameters();
        }


        void reset_parameters() {
            torch::nn::init::kaiming_uniform_(weight, std::sqrt(5)); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
            if (bias.defined()) {
                int64_t fan_in, fan_out;
                std::tie(fan_in, fan_out) =
                        torch::nn::init::_calculate_fan_in_and_fan_out(weight);
                const auto bound = 1 / std::sqrt(fan_in);
                torch::nn::init::uniform_(bias, -bound, bound);
            }
        }

        /// Pretty prints the `Linear` module into the given `stream`.
        void pretty_print(std::ostream &stream) const override {
            stream << "PIM::PimConv2d"
            << "(" << pim_type
            << ", " << options.in_channels()
            << ", " << options.out_channels()
            << ", kernel_size=" << options.kernel_size()
            << ", stride=" << options.stride();
            if (*options.padding() != *ExpandingArray<2>(0)) {
                stream << ", padding=" << options.padding();
            }
            stream << ", dilation=1";
            stream << ", output_padding=0";
            stream << ", groups=1";

            if (!options.bias()) {
                stream << ", bias=" << std::boolalpha << false;
            }
            if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
                stream << ", padding_mode=" << enumtype::get_enum_name(options.padding_mode());
            }
            stream << ")" << std::endl;
            if (print_detail_) {
                wb_ptr.ptr->print(stream);
                stream << "array_t" << std::endl;
                wb_t_ptr.ptr->print(stream);
                stream << "prevs" << std::endl;
                for (auto &i : prev_ptrs.ptrs) {
                    i->print(stream);
                }
            }

        }

        /// Transforms the `input` tensor by multiplying with the `weight` and
        /// optionally adding the `bias`, if `with_bias` is true in the options.
        Tensor forward(const Tensor &input) {
            switch (pim_type) {
                case PimArrayType::simple_logic_array:
                    return PimConv2dFunction<SimpleLogicArray>::apply(
                            wb_ptr, wb_t_ptr, prev_ptrs,
                            input, weight, options.bias() ? bias : c10::optional<Tensor>(),
                            options.stride(), options.padding(), is_training(), fast_mode);
                    case PimArrayType::pim_array_pro:
                        return PimConv2dFunction<pimArrayPro>::apply(
                                wb_ptr, wb_t_ptr, prev_ptrs,
                                input, weight, options.bias() ? bias : c10::optional<Tensor>(),
                                options.stride(), options.padding(), is_training(), fast_mode);
                        case PimArrayType::pim_array_fast:
                            return PimConv2dFunction<pimArrayFast>::apply(
                                    wb_ptr, wb_t_ptr, prev_ptrs,
                                    input, weight, options.bias() ? bias : c10::optional<Tensor>(),
                                    options.stride(), options.padding(), is_training(), fast_mode);
                            default:
                                TORCH_INTERNAL_ASSERT(false, "pim conv, forward type not supported!")
            }
        }

        void sync_weight() {
            // update parameters, note that weight shape is (Cin * H * W, Cout)
            if (bias.defined()) {
                wb_ptr.ptr->write_mat(torch::cat({
                    weight.permute({1, 2, 3, 0}).reshape({-1, options.out_channels()}),
                    bias.unsqueeze(0)}, 0));
                torch::Tensor w_idx = torch::arange(wb_ptr.ptr->sizes()[0]-1, TensorOptions(torch::kLong).device(weight.device()));
                weight.data() = wb_ptr.ptr->read_mat().index_select(0,w_idx).reshape({
                    weight.size(1), weight.size(2), weight.size(3), options.out_channels()}).permute({3, 0, 1, 2});
                bias.data() = wb_ptr.ptr->read_row(wb_ptr.ptr->sizes()[0]-1, 0, wb_ptr.ptr->sizes()[1]);
            } else {
                wb_ptr.ptr->write_mat(weight.permute({1, 2, 3, 0}).reshape({-1, options.out_channels()}));
                weight.data() = wb_ptr.ptr->read_mat().reshape({
                    weight.size(1), weight.size(2), weight.size(3), options.out_channels()}).permute({3, 0, 1, 2});
            }
            // flip kernel
            wb_t_ptr.ptr->write_mat(weight.flip({2, 3}).permute({1, 0, 2, 3}).reshape({options.in_channels(), -1}).t());
        }

        //    void train(bool on = true) override {
        //      if (!on) {
        //        sync_weight();
        //      }
        //      is_training_ = on;
        //    }

        void print_detail(bool on = true) {
            print_detail_ = on;
        }

        /// The options used to configure this module.
        Conv2dOptions options;

        /// The learned weight.
        Tensor weight;

        /// The learned bias. If `bias` is false in the `options`, this tensor is
        /// undefined.
        Tensor bias;

        //    /// Whether the module is in training mode.
        //    bool is_training_{true};

        /// Whether the physical array is be printed.
        bool print_detail_{false};

        bool fast_mode;

        PimArrayPtr wb_ptr;
        PimArrayPtr wb_t_ptr;
        PimArrayPtrList prev_ptrs;
        PimArrayType pim_type;
        ExpandingArray<4> input_shape;

    };

    TORCH_MODULE(PimConv2d);
}
#endif //PIMTORCH_PIM_CONV_H
