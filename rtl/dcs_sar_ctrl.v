`timescale 100ps/1ps

module ad_controller #(
    parameter resolution = 8, 
    digital_size = 3,
    state_size = 4,
    err_size = 3,

    //Ctrl States
    S_BEGIN = 4'b0000,
    S_FIRST = 4'b0001,
    S_CMN_1 = 4'b0010,
    S_CMN_2 = 4'b0011,
    S_CMB_1 = 4'b0100,
    S_CMB_2 = 4'b0101,
    S_BS = 4'b0110,
    S_IDLE = 4'b0111,

    //Error States
    NO_ERR = 3'b000,
    CMB_RB = 3'b001,
    CMB_FW = 3'b010,
    CMN_RB = 3'b011,
    CMN_FW = 3'b100

)(
    input clk,
    input rst_n,
    input enable,
    input cmp_flag,
    input [digital_size-1:0] ns_start,
    input [digital_size-1:0] noff_start,
    output reg [resolution-1:0] digital_code,
    output reg [err_size-1:0] err_state,
    output reg fin_flag
);

reg [digital_size-1:0] ns;
reg [digital_size-1:0] noff;
reg [digital_size-1:0] cur_bit;
reg [state_size-1:0] cur_state;

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        ns <= 0;
        noff <= 0;
        cur_bit <= 0;
        cur_state <= S_IDLE;
        digital_code <= 0;
        err_state <= NO_ERR;
        fin_flag <= 0;
    end
    else begin
        //$display("\ncur_state: 0x%h, digital code: 0x%h, cur_bit: 0x%h, cmp_flag: 0x%h\n", cur_state, digital_code, cur_bit, cmp_flag);
        case(cur_state)
        S_IDLE: begin
            if(enable) begin
                cur_state <= S_BEGIN;
                //digital_code <= digital_code + 1;
                err_state <= NO_ERR;
                fin_flag <= 0;
            end
        end
        S_BEGIN: begin
            ns <= ns_start;
            noff <= noff_start;
            cur_state <= S_FIRST;
            err_state <= NO_ERR;
            digital_code <= (1 << ns_start) - (1 << noff_start);
        end
        S_FIRST: begin
            if(cmp_flag)  begin
                cur_state <= S_CMN_1;
                digital_code <= 1 << ns;
            end
            else begin
                if(ns == noff + 1) begin
                    if(noff == 0) begin // 0 end
                        digital_code <= 0;
                        cur_state <= S_IDLE;
                        fin_flag <= 1;
                    end
                    else begin
                        cur_state <= S_CMB_2;
                        noff <= noff - 1;
                        digital_code <= 1 << (noff - 1);
                    end
                end
                else begin
                    cur_state <= S_CMN_2;
                    noff <= noff + 1;
                    digital_code <= (1 << ns) - (1 << (noff + 1));
                end
            end
        end
        S_CMN_1: begin
            if(cmp_flag) begin
                if(ns == resolution - 1) begin
                    cur_state <= S_BS;
                    cur_bit <= ns - 1;
                    digital_code <= (1 << ns) + (1 << (ns - 1));
                end
                else begin
                    cur_state <= S_CMB_1;
                    ns <= ns + 1;
                    digital_code <= 1 << (ns + 1);
                end
            end
            else begin
                if(noff == 0) begin // 1 end
                    digital_code <= (1 << ns) - 1;
                    cur_state <= S_IDLE;
                    fin_flag <= 1;
                end
                else begin
                    err_state <= CMN_FW;
                    cur_state <= S_BS;
                    noff <= noff - 1;
                    cur_bit <= noff - 1;
                    digital_code <= (1 << ns) - (1 << (noff - 1));
                end
            end
        end
        S_CMN_2: begin
            if(cmp_flag) begin
                if(noff == 1) begin // rare end, no error
                    cur_state <= S_IDLE;
                    digital_code <= (1 << ns) - (1 << noff);
                    fin_flag <= 1;
                end
                else begin
                    cur_state <= S_BS;
                    cur_bit <= noff - 2;
                    digital_code <= (1 << ns) - (1 << noff) + (1 << (noff - 2));
                end
            end
            else begin
                err_state <= CMN_RB;
                if(ns == noff + 1) begin
                    cur_state <= S_CMB_2;
                    noff <= noff - 1;
                    digital_code <= 1 << (noff - 1);
                end
                else begin
                    cur_state <= S_CMN_2;
                    noff <= noff + 1;
                    digital_code <= (1 << ns) - (1 << noff + 1);
                end
            end
        end
        S_CMB_1: begin
            err_state <= CMB_RB;
            if(cmp_flag) begin
                if(ns == resolution - 1)begin
                    cur_state <= S_BS;
                    cur_bit <= ns - 1;
                    digital_code = (1 << ns) + (1 << (ns - 1));
                end
                else begin
                    cur_state <= S_CMB_1;
                    ns <= ns + 1;
                    digital_code <= 1 << (ns + 1);
                end
            end
            else begin
                cur_state <= S_BS;
                cur_bit <= ns - 2;
                digital_code <= (1 << (ns - 2)) + (1 << (ns - 1));
            end
        end
        S_CMB_2: begin
            if(err_state == 0) begin
                err_state <= CMB_FW;
            end
            if(cmp_flag) begin
                if(noff == 0) begin  // 1 end
                    cur_state <= S_IDLE;
                    digital_code <= 1;
                    fin_flag <= 1;
                end
                else begin
                    cur_state <= S_BS;
                    digital_code = (1 << noff) + (1 << (noff - 1));
                    cur_bit <= noff - 1;
                end
            end
            else begin
                if(noff == 0) begin // 0 end
                    cur_state <= S_IDLE;
                    digital_code <= 0;
                    fin_flag <= 1;
                end
                else begin
                    cur_state <= S_BS;
                    digital_code = 1 << (noff - 1);
                    cur_bit <= noff - 1;
                end
            end
        end
        S_BS: begin
            if(cmp_flag) begin
                if(cur_bit == 0) begin
                    cur_state <= S_IDLE;
                    fin_flag <= 1;
                end
                else begin
                    cur_bit <= cur_bit - 1;
                    cur_state <= S_BS;
                    digital_code <= digital_code + (1 << (cur_bit - 1));
                end
            end
            else begin
                if(cur_bit == 0) begin
                    cur_state <= S_IDLE;
                    fin_flag <= 1;
                    digital_code <= digital_code - 1;
                end
                else begin
                    cur_bit <= cur_bit - 1;
                    cur_state <= S_BS;
                    digital_code <= digital_code - (1 << cur_bit) + (1 << (cur_bit - 1));
                end
            end
        end
        default: begin
            cur_state <= S_IDLE;
        end
        endcase
    end
end

endmodule

