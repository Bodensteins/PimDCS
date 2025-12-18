`timescale 100ps/1ps

module pseudo_cmp #(
    parameter resolution = 8
)(
    input clk,
    input rst_n,
    input [resolution-1:0] pseduo_vol_in,
    input [resolution-1:0] pseduo_vol_ref,
    output reg cmp_flag
);
    
always @(posedge clk, negedge rst_n) begin
    if(!rst_n) cmp_flag <= 0;
    else if(pseduo_vol_in < pseduo_vol_ref) cmp_flag <= 0;
    else cmp_flag <= 1;
end

endmodule
