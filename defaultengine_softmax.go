package tensor

import (
	"fmt"
	"math"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
)

// if dims = 2 and axis -1 it returns the last dimension. In this case 1
func resolveAxis(axis int, dims int) int {
	res := axis % dims
	if (res < 0 && dims > 0) || (res > 0 && dims < 0) {
		return res + dims
	}

	return res
}

func (e StdEng) LogSoftMax(x Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	axis = resolveAxis(axis, x.Dims())
	expectedShape := x.Shape().Clone()

	var reuse DenseTensor
	var safe, toReuse, _ bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(expectedShape, x.Dtype(), x.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if safe || !toReuse && reuse == nil && safe {
		// create reuse
		reuse = New(WithShape(expectedShape...), Of(x.Dtype()))
	}

	switch x.Dtype() {
	case Float32:
		if expectedShape.Dims()-1 == axis {
			e.logSoftMaxLastDimF32(reuse, x, axis)
		} else {
			e.softMaxInnerDimF32(reuse, x, axis, true)
		}
	case Float64:
		if expectedShape.Dims()-1 == axis {
			e.logSoftMaxLastDimF64(reuse, x, axis)
		} else {
			e.softMaxInnerDimF64(reuse, x, axis, true)
		}
	default:
		return nil, fmt.Errorf("type %v not supported", x.Dtype())
	}

	return reuse, nil
}

func (e StdEng) logSoftMaxLastDimF64(output Tensor, x Tensor, axis int) {
	outputArr := output.Data().([]float64)
	xArr := x.Data().([]float64)
	xShape := x.Shape()

	outerSize := 1
	dimSize := xShape[axis]
	for i := 0; i < axis; i++ {
		outerSize *= xShape[i]
	}

	for ii := 0; ii < outerSize; ii++ {
		maxInput := xArr[0]
		for j := 1; j < dimSize; j++ {
			i := ii*dimSize + j

			if xArr[i] > maxInput {
				maxInput = xArr[i]
			}
		}

		sumExp := 0.0
		for j := 0; j < dimSize; j++ {
			i := ii*dimSize + j

			outputArr[i] = xArr[i] - maxInput

			exp := math.Exp(outputArr[i])
			sumExp += exp
		}

		for j := 0; j < dimSize; j++ {
			i := ii*dimSize + j

			outputArr[i] -= math.Log(sumExp)
		}
	}
}

func (e StdEng) softMaxInnerDimF64(output Tensor, x Tensor, axis int, logSoftmax bool) {
	xShape := x.Shape()

	innerSize, outerSize := 1, 1
	for i := 0; i < axis; i++ {
		outerSize *= xShape[i]
	}

	for i := axis + 1; i < xShape.Dims(); i++ {
		innerSize *= xShape[i]
	}

	dimSize := xShape[axis]
	dimStride := innerSize
	outerStride := dimSize * dimStride

	outputArr := output.Data().([]float64)
	xArr := x.Data().([]float64)

	for ii := 0; ii < innerSize*outerSize; ii++ {
		outerIndex := ii / innerSize
		innerIndex := ii % innerSize

		inputPart := xArr[outerIndex*outerStride+innerIndex:]
		outputPart := outputArr[outerIndex*outerStride+innerIndex:]

		maxInput := xArr[0]
		for j := 1; j < dimSize; j++ {
			i := j * dimStride

			if inputPart[i] > maxInput {
				maxInput = inputPart[i]
			}
		}

		sumExp := 0.0
		for j := 0; j < dimSize; j++ {
			i := j * dimStride

			exp := math.Exp(inputPart[i] - maxInput)

			if !logSoftmax {
				outputPart[i] = exp
			}

			sumExp += exp
		}

		if logSoftmax {
			sumExp = math.Log(sumExp)
		} else {
			sumExp = 1 / sumExp
		}

		for j := 0; j < dimSize; j++ {
			i := j * dimStride

			if logSoftmax {
				outputPart[i] = inputPart[i] - maxInput - sumExp
			} else {
				outputPart[i] *= sumExp
			}
		}
	}
}

func (e StdEng) logSoftMaxLastDimF32(output Tensor, x Tensor, axis int) {
	outputArr := output.Data().([]float32)
	xArr := x.Data().([]float32)
	xShape := x.Shape()

	outerSize := 1
	dimSize := xShape[axis]
	for i := 0; i < axis; i++ {
		outerSize *= xShape[i]
	}

	for ii := 0; ii < outerSize; ii++ {
		maxInput := xArr[0]
		for j := 1; j < dimSize; j++ {
			i := ii*dimSize + j

			if xArr[i] > maxInput {
				maxInput = xArr[i]
			}
		}

		sumExp := float32(0.0)
		for j := 0; j < dimSize; j++ {
			i := ii*dimSize + j

			outputArr[i] = xArr[i] - maxInput

			exp := math32.Exp(outputArr[i])
			sumExp += exp
		}

		for j := 0; j < dimSize; j++ {
			i := ii*dimSize + j

			outputArr[i] -= math32.Log(sumExp)
		}
	}
}

func (e StdEng) softMaxInnerDimF32(output Tensor, x Tensor, axis int, logSoftmax bool) {
	xShape := x.Shape()

	innerSize, outerSize := 1, 1
	for i := 0; i < axis; i++ {
		outerSize *= xShape[i]
	}

	for i := axis + 1; i < xShape.Dims(); i++ {
		innerSize *= xShape[i]
	}

	dimSize := xShape[axis]
	dimStride := innerSize
	outerStride := dimSize * dimStride

	outputArr := output.Data().([]float32)
	xArr := x.Data().([]float32)

	for ii := 0; ii < innerSize*outerSize; ii++ {
		outerIndex := ii / innerSize
		innerIndex := ii % innerSize

		inputPart := xArr[outerIndex*outerStride+innerIndex:]
		outputPart := outputArr[outerIndex*outerStride+innerIndex:]

		maxInput := xArr[0]
		for j := 1; j < dimSize; j++ {
			i := j * dimStride

			if inputPart[i] > maxInput {
				maxInput = inputPart[i]
			}
		}

		sumExp := float32(0.0)
		for j := 0; j < dimSize; j++ {
			i := j * dimStride

			exp := math32.Exp(inputPart[i] - maxInput)

			if !logSoftmax {
				outputPart[i] = exp
			}

			sumExp += exp
		}

		if logSoftmax {
			sumExp = math32.Log(sumExp)
		} else {
			sumExp = 1 / sumExp
		}

		for j := 0; j < dimSize; j++ {
			i := j * dimStride

			if logSoftmax {
				outputPart[i] = inputPart[i] - maxInput - sumExp
			} else {
				outputPart[i] *= sumExp
			}
		}
	}
}
