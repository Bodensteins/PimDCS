import math

class ADC(object):
    def __init__(self, Vref, bit_width):
        self.Vref = Vref
        self.bit_width = bit_width
        self.deltaV = Vref / (1 << bit_width)

    def analog2digit(self, voltage):
        return int(voltage / self.deltaV)


class naiveADC(ADC):
    def __init__(self, row, Gmax, Gmin, cellBits, Vcom, R):
        Vref = row * Gmax * Vcom * R
        bit_width = math.ceil(math.log2(row)) + cellBits
        super.__init__(Vref, bit_width)


class myADC(ADC):
    def __init__(self, row, Gmax, Gmin, cellBits, Vcom, R):
        deltaG =
        deltaV = Vcom * R *
        bit_width = math.ceil(math.log2(row)) + cellBits
        super.__init__(Vref, bit_width)