`timescale 100ps/1ps

module adc #(
    parameter resolution = 8, 
    digital_size = 3
)(
    input clk,
    input rst_n,
    input enable,
    input [digital_size-1:0] ns_start,
    input [digital_size-1:0] noff_start,
    input [resolution-1:0] pseduo_vol_in,
    output [resolution-1:0] digital_output,
    output fin_flag
);

wire [resolution-1:0] reference;
wire cmp_flag;

assign digital_output = reference;

ad_controller u_ad_controller(
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .cmp_flag(cmp_flag),
    .ns_start(ns_start),
    .noff_start(noff_start),
    .digital_code(reference),
    .fin_flag(fin_flag)
);

pseudo_cmp u_pseudo_cmp(
    .clk(clk),
    .rst_n(rst_n),
    .pseduo_vol_in(pseduo_vol_in),
    .pseduo_vol_ref(reference),
    .cmp_flag(cmp_flag)
);

endmodule