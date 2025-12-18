set_app_var search_path  "/home/leitaoming/Project/Circuit/library/synopsys-lib-from-xujie"
set_app_var target_library "typical.db"
set_app_var link_library  "typical.db"

set_svf "./syn/synthesis.svf"
set sh_command_log_file "./syn/command.log"
set sh_output_log_file "./syn/synthesis_report"

read_verilog -f ./rtl/veri_src.f
current_design "adc"

check_design

create_clock -period 10 [get_ports clk]
set_input_delay -max 3 -clock clk [remove_from_collection [all_inputs] clk]
set_output_delay -max 2.5 -clock clk [all_outputs]
set_input_transition 0.15 [all_inputs]
set_host_options -max_cores 48

compile

# write -format verilog -hierarchy -output ./syn/netlist.v
# write_sdc ./syn/constraints.sdc

report_area > ./syn/area.rpt
report_power > ./syn/power.rpt
report_timing > ./syn/timing.rpt

quit