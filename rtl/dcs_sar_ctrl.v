`timescale 100ps/1ps

module ad_controller #(
    parameter resolution = 8, 
    digital_size = 3,
    state_size = 4,

    //Ctrl States
    BEG = 4'b0000,
    FIRST_CMP = 4'b0001,
    GREATER = 4'b0010,
    LESS_NORM1 = 4'b0011,
    LESS_NORM2 = 4'b0100,
    LESS_MONO = 4'b0101,
    GREATER_ERR1 = 4'b0110,
    GREATER_ERR2 = 4'b0111,
    BS1 = 4'b1000,
    BS2 = 4'b1001,
    IDLE = 4'b1010

    //Error States
    // NO_ERR = 3'b000
    // NS_LEFT = 3'b001
    // NS_RIGHT = 3'b010
    // NOFF_LEFT = 3'b011
    // NOFF_RIGHT= 3'b100

)(
    input clk,
    input rst_n,
    input enable,
    input cmp_flag,
    input [digital_size-1:0] ns_start,
    input [digital_size-1:0] noff_start,
    // input [digital_size-1:0] err_state,
    output reg [resolution-1:0] digital_code,
    output reg fin_flag
);

reg [digital_size-1:0] ns;
reg [digital_size-1:0] noff;
reg [digital_size-1:0] cur_bit;
reg [state_size-1:0] cur_state;
reg [state_size-1:0] next_state;

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        ns <= 0;
        noff <= 0;
        cur_bit <= 0;
        cur_state <= IDLE;
        digital_code <= 0;
        fin_flag <= 0;
    end
    else begin
        cur_state <= next_state;
    end
end

always @(cur_state, enable) begin
    //$display("\ncur_state: 0x%h, digital code: 0x%h, cur_bit: 0x%h, cmp_flag: 0x%h\n", cur_state, digital_code, cur_bit, cmp_flag);
    case(cur_state)
    IDLE: begin
        if(enable) begin
            next_state <= BEG;
            fin_flag <= 0;
        end
    end
    BEG: begin
        ns <= ns_start;
        noff <= noff_start;
        next_state <= FIRST_CMP;
        digital_code <= (1 << ns_start) - (1 << noff_start);
    end
    FIRST_CMP: begin
        if(cmp_flag)  begin
            next_state <= GREATER;
            digital_code <= 1 << ns;
        end
        else begin
            if(ns == noff + 1) begin
                if(noff == 0) begin // 0 end
                    digital_code <= 0;
                    next_state <= IDLE;
                    fin_flag <= 1;
                end
                else begin
                    next_state <= LESS_MONO;
                    noff <= noff - 1;
                    digital_code <= 1 << (noff - 1);
                end
            end
            else begin
                next_state <= LESS_NORM1;
                noff <= noff + 1;
                digital_code <= (1 << ns) - (1 << (noff + 1));
            end
        end
    end
    GREATER: begin
        if(cmp_flag) begin
            if(ns == resolution - 1) begin
                next_state <= BS1;
                cur_bit <= ns - 1;
                digital_code <= (1 << ns) + (1 << (ns - 1));
            end
            else begin
                next_state <= GREATER_ERR1;
                ns <= ns + 1;
                digital_code <= 1 << (ns + 1);
            end
        end
        else begin
            if(noff == 0) begin // 1 end
                digital_code <= (1 << ns) - 1;
                next_state <= IDLE;
                fin_flag <= 1;
            end
            else begin
                next_state <= BS1;
                noff <= noff - 1;
                cur_bit <= noff - 1;
                digital_code <= (1 << ns) - (1 << (noff - 1));
            end
        end
    end
    LESS_NORM1: begin    //LESS_NROM1, LESS_NORM2
        if(cmp_flag) begin
            if(noff == 1) begin // rare end
                next_state <= IDLE;
                digital_code <= (1 << ns) - (1 << noff);
                fin_flag <= 1;
            end
            else begin
                next_state <= BS1;
                cur_bit <= noff - 2;
                digital_code <= (1 << ns) - (1 << noff) + (1 << (noff - 2));
            end
        end
        else begin
            if(ns == noff + 1) begin
                next_state <= LESS_MONO;
                noff <= noff - 1;
                digital_code <= 1 << (noff - 1);
            end
            else begin
                next_state <= LESS_NORM2;
                noff <= noff + 1;
                digital_code <= (1 << ns) - (1 << noff + 1);
            end
        end
    end
    LESS_NORM2: begin    //LESS_NROM1, LESS_NORM2
        if(cmp_flag) begin
            if(noff == 1) begin // rare end
                next_state <= IDLE;
                digital_code <= (1 << ns) - (1 << noff);
                fin_flag <= 1;
            end
            else begin
                next_state <= BS1;
                cur_bit <= noff - 2;
                digital_code <= (1 << ns) - (1 << noff) + (1 << (noff - 2));
            end
        end
        else begin
            if(ns == noff + 1) begin
                next_state <= LESS_MONO;
                noff <= noff - 1;
                digital_code <= 1 << (noff - 1);
            end
            else begin
                next_state <= LESS_NORM1;
                noff <= noff + 1;
                digital_code <= (1 << ns) - (1 << noff + 1);
            end
        end
    end
    LESS_MONO: begin
        if(cmp_flag) begin
            if(noff == 0)begin  // 1 end
                next_state <= IDLE;
                digital_code <= 1;
                fin_flag <= 1;
            end
            else begin
                next_state <= BS1;
                digital_code = (1 << noff) + (1 << (noff - 1));
                cur_bit <= noff - 1;
            end
        end
        else begin
            if(noff == 0) begin // 0 end
                next_state <= IDLE;
                digital_code <= 0;
                fin_flag <= 1;
            end
            else begin
                next_state <= BS1;
                digital_code = 1 << (noff - 1);
                cur_bit <= noff - 1;
            end
        end
    end
    GREATER_ERR1: begin  //GREATER_ERR1, GREATER_ERR2
        if(cmp_flag) begin
            if(ns == resolution - 1)begin
                next_state <= BS1;
                cur_bit <= ns - 1;
                digital_code = (1 << ns) + (1 << (ns - 1));
            end
            else begin
                next_state <= GREATER_ERR2;
                ns <= ns + 1;
                digital_code <= 1 << (ns + 1);
            end
        end
        else begin
            next_state <= BS1;
            cur_bit <= ns - 2;
            digital_code <= (1 << (ns - 2)) + (1 << (ns - 1));
        end
    end
    GREATER_ERR2: begin  //GREATER_ERR1, GREATER_ERR2
        if(cmp_flag) begin
            if(ns == resolution - 1)begin
                next_state <= BS1;
                cur_bit <= ns - 1;
                digital_code = (1 << ns) + (1 << (ns - 1));
            end
            else begin
                next_state <= GREATER_ERR1;
                ns <= ns + 1;
                digital_code <= 1 << (ns + 1);
            end
        end
        else begin
            next_state <= BS1;
            cur_bit <= ns - 2;
            digital_code <= (1 << (ns - 2)) + (1 << (ns - 1));
        end
    end
    BS1: begin
        if(cmp_flag) begin
            if(cur_bit == 0) begin
                next_state <= IDLE;
                fin_flag <= 1;
            end
            else begin
                cur_bit <= cur_bit - 1;
                next_state <= BS2;
                digital_code <= digital_code + (1 << (cur_bit - 1));
            end
        end
        else begin
            if(cur_bit == 0) begin
                next_state <= IDLE;
                fin_flag <= 1;
                digital_code <= digital_code - 1;
            end
            else begin
                cur_bit <= cur_bit - 1;
                next_state <= BS2;
                digital_code <= digital_code - (1 << cur_bit) + (1 << (cur_bit - 1));
            end
        end
    end
    BS2: begin
        if(cmp_flag) begin
            if(cur_bit == 0) begin
                next_state <= IDLE;
                fin_flag <= 1;
            end
            else begin
                cur_bit <= cur_bit - 1;
                next_state <= BS1;
                digital_code <= digital_code + (1 << (cur_bit - 1));
            end
        end
        else begin
            
            if(cur_bit == 0) begin
                next_state <= IDLE;
                fin_flag <= 1;
                digital_code <= digital_code - 1;
            end
            else begin
                cur_bit <= cur_bit - 1;
                next_state <= BS1;
                digital_code <= digital_code - (1 << cur_bit) + (1 << (cur_bit - 1));
            end
        end
    end
    default: begin
        next_state <= IDLE;
    end
    endcase
end

endmodule

