`timescale 100ps/1ps

module pseudo_cmp #(
    parameter resolution = 8
)(
    input [resolution-1:0] pseduo_vol_in,
    input [resolution-1:0] pseduo_vol_ref,
    output cmp_flag
);

assign cmp_flag = (pseduo_vol_in >= pseduo_vol_ref) ? 1'b1 : 1'b0;

endmodule
