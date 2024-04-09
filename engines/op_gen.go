// Code generated by genlib3. DO NOT EDIT

package stdeng

import (
	"golang.org/x/exp/constraints"
	"gorgonia.org/tensor/internal/execution"
)

// ltOp creates a `CmpBinOp` for values of the `constraints.Ordered` datatype.
func ltOp[DT constraints.Ordered]() CmpBinOp[DT] {
	return CmpBinOp[DT]{
		VV:     execution.LtVVBool[DT],
		VVBC:   execution.LtBCBool[DT],
		VVIter: execution.LtVVIterBool[DT],

		VS:     execution.LtVSBool[DT],
		VSIter: execution.LtVSIterBool[DT],

		SV:     execution.LtSVBool[DT],
		SVIter: execution.LtSVIterBool[DT],
	}
}

// lteOp creates a `CmpBinOp` for values of the `constraints.Ordered` datatype.
func lteOp[DT constraints.Ordered]() CmpBinOp[DT] {
	return CmpBinOp[DT]{
		VV:     execution.LteVVBool[DT],
		VVBC:   execution.LteBCBool[DT],
		VVIter: execution.LteVVIterBool[DT],

		VS:     execution.LteVSBool[DT],
		VSIter: execution.LteVSIterBool[DT],

		SV:     execution.LteSVBool[DT],
		SVIter: execution.LteSVIterBool[DT],
	}
}

// gtOp creates a `CmpBinOp` for values of the `constraints.Ordered` datatype.
func gtOp[DT constraints.Ordered]() CmpBinOp[DT] {
	return CmpBinOp[DT]{
		VV:     execution.GtVVBool[DT],
		VVBC:   execution.GtBCBool[DT],
		VVIter: execution.GtVVIterBool[DT],

		VS:     execution.GtVSBool[DT],
		VSIter: execution.GtVSIterBool[DT],

		SV:     execution.GtSVBool[DT],
		SVIter: execution.GtSVIterBool[DT],
	}
}

// gteOp creates a `CmpBinOp` for values of the `constraints.Ordered` datatype.
func gteOp[DT constraints.Ordered]() CmpBinOp[DT] {
	return CmpBinOp[DT]{
		VV:     execution.GteVVBool[DT],
		VVBC:   execution.GteBCBool[DT],
		VVIter: execution.GteVVIterBool[DT],

		VS:     execution.GteVSBool[DT],
		VSIter: execution.GteVSIterBool[DT],

		SV:     execution.GteSVBool[DT],
		SVIter: execution.GteSVIterBool[DT],
	}
}

// eleqOp creates a `CmpBinOp` for values of the `comparable` datatype.
func eleqOp[DT comparable]() CmpBinOp[DT] {
	return CmpBinOp[DT]{
		VV:     execution.ElEqVVBool[DT],
		VVBC:   execution.ElEqBCBool[DT],
		VVIter: execution.ElEqVVIterBool[DT],

		VS:     execution.ElEqVSBool[DT],
		VSIter: execution.ElEqVSIterBool[DT],

		SV:     execution.ElEqSVBool[DT],
		SVIter: execution.ElEqSVIterBool[DT],
	}
}

// elneOp creates a `CmpBinOp` for values of the `comparable` datatype.
func elneOp[DT comparable]() CmpBinOp[DT] {
	return CmpBinOp[DT]{
		VV:     execution.ElNeVVBool[DT],
		VVBC:   execution.ElNeBCBool[DT],
		VVIter: execution.ElNeVVIterBool[DT],

		VS:     execution.ElNeVSBool[DT],
		VSIter: execution.ElNeVSIterBool[DT],

		SV:     execution.ElNeSVBool[DT],
		SVIter: execution.ElNeSVIterBool[DT],
	}
}

// ltOpOrderedNum creates the ops necessary for an OrderedNum engine.
func ltOpOrderedNum[DT OrderedNum]() (Op[DT], CmpBinOp[DT]) {
	return Op[DT]{
		VV:     execution.LtVV[DT],
		VVBC:   execution.LtBC[DT],
		VVIter: execution.LtVVIter[DT],

		VS:     execution.LtVS[DT],
		VSIter: execution.LtVSIter[DT],

		SV:     execution.LtSV[DT],
		SVIter: execution.LtSVIter[DT],
	}, ltOp[DT]()
}

// lteOpOrderedNum creates the ops necessary for an OrderedNum engine.
func lteOpOrderedNum[DT OrderedNum]() (Op[DT], CmpBinOp[DT]) {
	return Op[DT]{
		VV:     execution.LteVV[DT],
		VVBC:   execution.LteBC[DT],
		VVIter: execution.LteVVIter[DT],

		VS:     execution.LteVS[DT],
		VSIter: execution.LteVSIter[DT],

		SV:     execution.LteSV[DT],
		SVIter: execution.LteSVIter[DT],
	}, lteOp[DT]()
}

// gtOpOrderedNum creates the ops necessary for an OrderedNum engine.
func gtOpOrderedNum[DT OrderedNum]() (Op[DT], CmpBinOp[DT]) {
	return Op[DT]{
		VV:     execution.GtVV[DT],
		VVBC:   execution.GtBC[DT],
		VVIter: execution.GtVVIter[DT],

		VS:     execution.GtVS[DT],
		VSIter: execution.GtVSIter[DT],

		SV:     execution.GtSV[DT],
		SVIter: execution.GtSVIter[DT],
	}, gtOp[DT]()
}

// gteOpOrderedNum creates the ops necessary for an OrderedNum engine.
func gteOpOrderedNum[DT OrderedNum]() (Op[DT], CmpBinOp[DT]) {
	return Op[DT]{
		VV:     execution.GteVV[DT],
		VVBC:   execution.GteBC[DT],
		VVIter: execution.GteVVIter[DT],

		VS:     execution.GteVS[DT],
		VSIter: execution.GteVSIter[DT],

		SV:     execution.GteSV[DT],
		SVIter: execution.GteSVIter[DT],
	}, gteOp[DT]()
}

// eleqOpOrderedNum creates the ops necessary for an OrderedNum engine.
func eleqOpOrderedNum[DT Num]() (Op[DT], CmpBinOp[DT]) {
	return Op[DT]{
		VV:     execution.ElEqVV[DT],
		VVBC:   execution.ElEqBC[DT],
		VVIter: execution.ElEqVVIter[DT],

		VS:     execution.ElEqVS[DT],
		VSIter: execution.ElEqVSIter[DT],

		SV:     execution.ElEqSV[DT],
		SVIter: execution.ElEqSVIter[DT],
	}, eleqOp[DT]()
}

// elneOpOrderedNum creates the ops necessary for an OrderedNum engine.
func elneOpOrderedNum[DT Num]() (Op[DT], CmpBinOp[DT]) {
	return Op[DT]{
		VV:     execution.ElNeVV[DT],
		VVBC:   execution.ElNeBC[DT],
		VVIter: execution.ElNeVVIter[DT],

		VS:     execution.ElNeVS[DT],
		VSIter: execution.ElNeVSIter[DT],

		SV:     execution.ElNeSV[DT],
		SVIter: execution.ElNeSVIter[DT],
	}, elneOp[DT]()
}