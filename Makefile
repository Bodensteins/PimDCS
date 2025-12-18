all: compile simulate wave

compile:
	vcs -full64 \
		-debug_acc+all \
		-sverilog \
		-f ./rtl/veri_src.f \
		-l ./syn/vcs_compile.log

simulate:
	./simv -l ./syn/vcs_simulation.log

wave:
	verdi -f ./rtl/veri_src.f \
		-nologo \
		-ssf ./wave/wave.fsdb 

synthesis:
	dc_shell-t -f ./rtl/dc_synthesis.tcl

clean:
	@rm -rf csrc DVEfiles verdiLog ucli.key
	@rm -rf simv simv.daidir ./wave/* ./syn/*
	@rm -rf *.svf *.conf *.rc *.fsdb *.log

.PHONY: all compile simulate wave synthesis clean
