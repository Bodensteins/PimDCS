`timescale 100ps/1ps

module err_controller #(
    parameter resolution = 8, 
    err_size = 3,
    digital_size = 3,
    cnt_size = 2,
    err_fsm_size = 2,

    //Error FSM States
    E_Rst = 2'b00,
    E_Accum = 2'b01,
    E_Act = 2'b10,

    //Error States
    NO_ERR = 3'b000,
    CMB_RB = 3'b001,
    CMB_FW = 3'b010,
    CMN_RB = 3'b011,
    CMN_FW = 3'b100

)(
    input clk,
    input rst_n,
    input para_ini,
    input enable,
    input fin_flag,
    input [err_size-1:0] err_state,
    input [cnt_size-1:0] err_thres,
    input [digital_size-1:0] ns_ini,
    input [digital_size-1:0] noff_ini,
    output reg [digital_size-1:0] ns_start,
    output reg [digital_size-1:0] noff_start
);

reg [cnt_size-1:0] err_cnt;
reg [err_size-1:0] pre_err_state;
reg [err_fsm_size] cur_fsm_state;
reg [err_fsm_size] next_fsm_state;

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        err_cnt <= 0;
        pre_err_state <= NO_ERR;
        cur_fsm_state <= E_Rst;
        ns_start <= 0;
        noff_start <= 0;
    end
    else if(para_ini) begin
        err_cnt <= 0;
        pre_err_state <= NO_ERR;
        cur_fsm_state <= E_Rst;
        ns_start <= ns_ini;
        noff_start <= noff_ini;
    end
    else begin
        if(fin_flag) begin
            case(cur_state)
            E_Rst: begin
                if(enable and err_state != NO_ERR) begin
                    err_cnt <= 1;
                    next_state <= E_Accum;
                    pre_err_state <= err_state;
                end
            end
            E_Accum: begin
                if(err_state != pre_err_state) begin
                    err_cnt <= 0;
                    next_state <= E_Rst;
                end
                else if(err_cnt < err_thres - 1) begin
                    err_cnt <= err_cnt + 1;
                    next_state <= E_Accum;
                end
                else begin
                    err_cnt <= err_cnt + 1;
                    next_state <= E_Act;
                end
            end
            E_Act: begin
                err_cnt <= 0;
                next_state < E_Rst;
                if(err_state == pre_err_state) begin
                    case(err_state)
                    CMB_RB: begin
                        noff_start <= ns_start;
                        ns_start <= ns_start + 1;
                    end
                    CMB_FW: begin
                        ns_start <= noff_start;
                        noff_start <= noff_start - 1;
                    end
                    CMN_RB: begin
                        noff_start <= noff_start + 1;
                    end
                    CMN_FW: begin
                        noff_start <= noff_start - 1;
                    end
                    endcase
                end
            end
            endcase
        end
    end
end

endmodule