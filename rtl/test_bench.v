`timescale 100ps/1ps

module adc_tb #(
    parameter resolution = 8, 
    digital_size = 3,
    state_size = 4,
    err_size = 3
);

reg clk;
reg rst_n;
reg enable;
reg [digital_size-1:0] ns_start;
reg [digital_size-1:0] noff_start;
reg [resolution-1:0] pseduo_vol_in;
wire [resolution-1:0] digital_output;
wire fin_flag;
wire [digital_size-1:0] ns;
wire [digital_size-1:0] noff;
wire [state_size-1:0] ctrl_state;
wire [digital_size-1:0] cur_bit;
wire [err_size-1:0] err_state;
wire cmp_flag;

adc u_adc(
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .ns_start(ns_start),
    .noff_start(noff_start),
    .pseduo_vol_in(pseduo_vol_in),
    .digital_output(digital_output),
    .err_state(err_state),
    .fin_flag(fin_flag)
);

assign ns = u_adc.u_ad_controller.ns;
assign noff = u_adc.u_ad_controller.noff;
assign ctrl_state = u_adc.u_ad_controller.cur_state;
assign cur_bit = u_adc.u_ad_controller.cur_bit;
assign cmp_flag = u_adc.u_ad_controller.cmp_flag;

initial begin
    #20000;
    $finish;
end

initial begin
    rst_n = 0;
    #20;
    rst_n = 1;
end

// cycle 1ns
initial begin
    clk = 0;
    forever begin
        #5 clk = ~clk;
    end
end

initial begin
    enable = 1;
    ns_start = 5;
    noff_start = 3;
    pseduo_vol_in = 26;

    #160
    enable = 1;
    ns_start = 1;
    noff_start = 0;
    pseduo_vol_in = 0;

    #160
    enable = 1;
    ns_start = 1;
    noff_start = 0;
    pseduo_vol_in = 1;

    #160
    enable = 1;
    ns_start = 1;
    noff_start = 0;
    pseduo_vol_in = 2;

    #160
    enable = 1;
    ns_start = 5;
    noff_start = 3;
    pseduo_vol_in = 26;

    #160
    enable = 1;
    ns_start = 5;
    noff_start = 3;
    pseduo_vol_in = 29;

    #160
    enable = 1;
    ns_start = 5;
    noff_start = 3;
    pseduo_vol_in = 28;

    #160
    enable = 1;
    ns_start = 5;
    noff_start = 3;
    pseduo_vol_in = 17;

    #160
    enable = 1;
    ns_start = 5;
    noff_start = 3;
    pseduo_vol_in = 4;

    #160;
    enable = 1;
    ns_start = 3;
    noff_start = 2;
    pseduo_vol_in = 1;

    #160
    enable = 1;
    ns_start = 5;
    noff_start = 2;
    pseduo_vol_in = 22;

    #160
    enable = 1;
    ns_start = 5;
    noff_start = 2;
    pseduo_vol_in = 36;

    #160
    enable = 1;
    ns_start = 5;
    noff_start = 2;
    pseduo_vol_in = 111;

    #160
    enable = 1;
    ns_start = 5;
    noff_start = 2;
    pseduo_vol_in = 245;

    #160
    enable = 1;
    ns_start = 4;
    noff_start = 3;
    pseduo_vol_in = 17;

    #160
    enable = 1;
    ns_start = 4;
    noff_start = 3;
    pseduo_vol_in = 14;

    #160
    enable = 1;
    ns_start = 4;
    noff_start = 3;
    pseduo_vol_in = 11;

    #160
    enable = 1;
    ns_start = 4;
    noff_start = 3;
    pseduo_vol_in = 7;

    #160
    enable = 1;
    ns_start = 4;
    noff_start = 3;
    pseduo_vol_in = 1;

    #160
    enable = 1;
    ns_start = 4;
    noff_start = 3;
    pseduo_vol_in = 12;

    #160
    enable = 1;
    ns_start = 7;
    noff_start = 6;
    pseduo_vol_in = 63;

    #160
    enable = 1;
    ns_start = 7;
    noff_start = 6;
    pseduo_vol_in = 221;

    #160
    enable = 1;
    ns_start = 7;
    noff_start = 6;
    pseduo_vol_in = 4;

end

initial begin
    $fsdbDumpfile("wave/wave.fsdb");
    $fsdbDumpvars(0, adc_tb);
end

endmodule
