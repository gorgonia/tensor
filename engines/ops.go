package stdeng

import (
	"gorgonia.org/tensor/internal/execution"
)

type vvFunc[DTa, DTb, DTc any] func(a []DTa, b []DTb, c []DTc)

type vvIterFunc[DTa, DTb, DTc any] func(a []DTa, b []DTb, c []DTc, ait, bit, cit Iterator) error

type vsFunc[DTa, DTb, DTc any] func(a []DTa, b DTb, c []DTc)

type svFunc[DTa, DTb, DTc any] func(a DTa, b []DTb, c []DTc)

type vsIterFunc[DTa, DTb, DTc any] func(a []DTa, b DTb, c []DTc, ait, cit Iterator) error

type svIterFunc[DTa, DTb, DTc any] func(a DTa, b []DTb, c []DTc, bit, cit Iterator) error

// Op represents an operation. It's basically a tuple of functions
type Op[DT any] struct {
	VV, VVIncr         vvFunc[DT, DT, DT]
	VVIter, VVIncrIter vvIterFunc[DT, DT, DT]

	VS, VSIncr         vsFunc[DT, DT, DT]
	VSIter, VSIncrIter vsIterFunc[DT, DT, DT]

	SV, SVIncr         svFunc[DT, DT, DT]
	SVIter, SVIncrIter svIterFunc[DT, DT, DT]
}

type CmpBinOp[DT any] struct {
	VV     vvFunc[DT, DT, bool]
	VVIter vvIterFunc[DT, DT, bool]

	VS     vsFunc[DT, DT, bool]
	VSIter vsIterFunc[DT, DT, bool]

	SV     svFunc[DT, DT, bool]
	SVIter svIterFunc[DT, DT, bool]
}

func addOp[DT Addable]() Op[DT] {
	return Op[DT]{
		VV:         execution.AddVV[DT],
		VVIncr:     execution.AddVVIncr[DT],
		VVIter:     execution.AddVVIter[DT],
		VVIncrIter: execution.AddVVIncrIter[DT],

		VS:         execution.AddVS[DT],
		VSIncr:     execution.AddVSIncr[DT],
		VSIter:     execution.AddVSIter[DT],
		VSIncrIter: execution.AddVSIncrIter[DT],

		SV:         execution.AddSV[DT],
		SVIncr:     execution.AddSVIncr[DT],
		SVIter:     execution.AddSVIter[DT],
		SVIncrIter: execution.AddSVIncrIter[DT],
	}
}

func subOp[DT Num]() Op[DT] {
	return Op[DT]{
		VV:         execution.SubVV[DT],
		VVIncr:     execution.SubVVIncr[DT],
		VVIter:     execution.SubVVIter[DT],
		VVIncrIter: execution.SubVVIncrIter[DT],

		VS:         execution.SubVS[DT],
		VSIncr:     execution.SubVSIncr[DT],
		VSIter:     execution.SubVSIter[DT],
		VSIncrIter: execution.SubVSIncrIter[DT],

		SV:         execution.SubSV[DT],
		SVIncr:     execution.SubSVIncr[DT],
		SVIter:     execution.SubSVIter[DT],
		SVIncrIter: execution.SubSVIncrIter[DT],
	}
}

func mulOp[DT Num]() Op[DT] {
	return Op[DT]{
		VV:         execution.MulVV[DT],
		VVIncr:     execution.MulVVIncr[DT],
		VVIter:     execution.MulVVIter[DT],
		VVIncrIter: execution.MulVVIncrIter[DT],

		VS:         execution.MulVS[DT],
		VSIncr:     execution.MulVSIncr[DT],
		VSIter:     execution.MulVSIter[DT],
		VSIncrIter: execution.MulVSIncrIter[DT],

		SV:         execution.MulSV[DT],
		SVIncr:     execution.MulSVIncr[DT],
		SVIter:     execution.MulSVIter[DT],
		SVIncrIter: execution.MulSVIncrIter[DT],
	}
}

func divOp[DT Num]() Op[DT] {
	return Op[DT]{
		VV:         execution.DivVV[DT],
		VVIncr:     execution.DivVVIncr[DT],
		VVIter:     execution.DivVVIter[DT],
		VVIncrIter: execution.DivVVIncrIter[DT],

		VS:         execution.DivVS[DT],
		VSIncr:     execution.DivVSIncr[DT],
		VSIter:     execution.DivVSIter[DT],
		VSIncrIter: execution.DivVSIncrIter[DT],

		SV:         execution.DivSV[DT],
		SVIncr:     execution.DivSVIncr[DT],
		SVIter:     execution.DivSVIter[DT],
		SVIncrIter: execution.DivSVIncrIter[DT],
	}
}

func minOp[DT OrderedNum]() Op[DT] {
	return Op[DT]{
		VV:     execution.MinVV[DT],
		VVIter: execution.MinVVIter[DT],

		VS:     execution.MinVS[DT],
		VSIter: execution.MinVSIter[DT],

		SV:     execution.MinSV[DT],
		SVIter: execution.MinSVIter[DT],
	}
}

func maxOp[DT OrderedNum]() Op[DT] {
	return Op[DT]{
		VV:     execution.MaxVV[DT],
		VVIter: execution.MaxVVIter[DT],

		VS:     execution.MaxVS[DT],
		VSIter: execution.MaxVSIter[DT],

		SV:     execution.MaxSV[DT],
		SVIter: execution.MaxSVIter[DT],
	}
}
