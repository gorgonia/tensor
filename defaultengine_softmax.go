package tensor

import (
	"fmt"
	"log"
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

func (e StdEng) SoftMax(x Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
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
			e.softMaxLastDimF32(reuse, x, axis, false)
		} else {
			e.softMaxInnerDimF32(reuse, x, axis, false)
		}
	case Float64:
		if expectedShape.Dims()-1 == axis {
			e.softMaxLastDimF64(reuse, x, axis, false)
		} else {
			e.softMaxInnerDimF64(reuse, x, axis, false)
		}
	default:
		return nil, fmt.Errorf("type %v not supported", x.Dtype())
	}

	return reuse, nil
}

func (e StdEng) SoftMaxB(output, grad Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	if !output.Shape().Eq(grad.Shape()) {
		return nil, fmt.Errorf("output and grad shapes don't match")
	}

	if !output.Dtype().Eq(grad.Dtype()) {
		return nil, fmt.Errorf("output and grad types don't match")
	}

	axis = resolveAxis(axis, output.Dims())
	expectedShape := output.Shape().Clone()

	var reuse DenseTensor
	var safe, toReuse, _ bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(expectedShape, output.Dtype(), output.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if safe || !toReuse && reuse == nil && safe {
		// create reuse
		reuse = New(WithShape(expectedShape...), Of(output.Dtype()))
	}

	switch output.Dtype() {
	case Float32:
		// if expectedShape.Dims()-1 == axis {
		// 	e.softMaxBLastDimF32(reuse, output, grad, axis, false)
		// } else {
		// 	e.softMaxBInnerDimF32(reuse, output, grad, axis, false)
		// }
		panic("float32 not implemented yet")
	case Float64:
		if expectedShape.Dims()-1 == axis {
			e.softMaxBLastDimF64(reuse, output, grad, axis, false)
		} else {
			// e.softMaxBInnerDimF64(reuse, output, grad, axis, false)
		}
	default:
		return nil, fmt.Errorf("type %v not supported", output.Dtype())
	}

	return reuse, nil
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
			e.softMaxLastDimF32(reuse, x, axis, true)
		} else {
			e.softMaxInnerDimF32(reuse, x, axis, true)
		}
	case Float64:
		if expectedShape.Dims()-1 == axis {
			e.softMaxLastDimF64(reuse, x, axis, true)
		} else {
			e.softMaxInnerDimF64(reuse, x, axis, true)
		}
	default:
		return nil, fmt.Errorf("type %v not supported", x.Dtype())
	}

	return reuse, nil
}

func (e StdEng) LogSoftMaxB(output, grad Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	if !output.Shape().Eq(grad.Shape()) {
		return nil, fmt.Errorf("output and grad shapes don't match")
	}

	if !output.Dtype().Eq(grad.Dtype()) {
		return nil, fmt.Errorf("output and grad types don't match")
	}

	axis = resolveAxis(axis, output.Dims())
	expectedShape := output.Shape().Clone()

	var reuse DenseTensor
	var safe, toReuse, _ bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(expectedShape, output.Dtype(), output.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if safe || !toReuse && reuse == nil && safe {
		// create reuse
		reuse = New(WithShape(expectedShape...), Of(output.Dtype()))
	}

	switch output.Dtype() {
	case Float32:
		// if expectedShape.Dims()-1 == axis {
		// 	e.softMaxLastBDimF32(reuse, output, grad, axis, true)
		// } else {
		// 	e.softMaxInnerBDimF32(reuse, output, grad, axis, true)
		// }
		panic("float32 not implemented yet")
	case Float64:
		if expectedShape.Dims()-1 == axis {
			e.softMaxBLastDimF64(reuse, output, grad, axis, true)
		} else {
			// e.softMaxBInnerDimF64(reuse, output, grad, axis, true)
		}
	default:
		return nil, fmt.Errorf("type %v not supported", output.Dtype())
	}

	return reuse, nil
}

func (e StdEng) softMaxLastDimF64(output Tensor, x Tensor, axis int, logSoftMax bool) {
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

		sumExp := float64(0.0)
		for j := 0; j < dimSize; j++ {
			i := ii*dimSize + j
			z := xArr[i] - maxInput
			exp := math.Exp(z)

			if logSoftMax {
				outputArr[i] = z
			} else {
				outputArr[i] = exp
			}

			sumExp += exp
		}

		if !logSoftMax {
			sumExp = 1 / sumExp
		}

		for j := 0; j < dimSize; j++ {
			i := ii*dimSize + j

			if logSoftMax {
				outputArr[i] -= math.Log(sumExp)
			} else {
				outputArr[i] *= sumExp
			}
		}
	}
}

func (e StdEng) softMaxBLastDimF64(inputGrad, output, grad Tensor, axis int, logSoftMax bool) {
	dx := inputGrad.Data().([]float64)
	outputArr := output.Data().([]float64)
	gradArr := grad.Data().([]float64)

	outputShape := output.Shape()

	outerSize := 1
	dimSize := outputShape[axis]
	for i := 0; i < axis; i++ {
		outerSize *= outputShape[i]
	}

	for ii := 0; ii < outerSize; ii++ {
		if logSoftMax {
			sum := gradArr[ii*dimSize]
			for j := 1; j < dimSize; j++ {
				i := ii*dimSize + j

				sum += gradArr[i]
			}

			log.Printf("sum: %v", sum)

			for j := 0; j < dimSize; j++ {
				i := ii*dimSize + j

				dx[i] = gradArr[i] - (math.Exp(outputArr[i]) * sum)
			}
		} else {
			mul := make([]float64, dimSize) // TODO: avoid extra alloc

			for j := 0; j < dimSize; j++ {
				i := ii*dimSize + j

				mul[j] = outputArr[i] * gradArr[i]
			}

			log.Printf("mul: %v", mul)

			sum := mul[0]
			for j := 1; j < dimSize; j++ {
				sum += mul[j]
			}

			for j := 0; j < dimSize; j++ {
				i := ii*dimSize + j

				dx[i] = (gradArr[i] - sum) * outputArr[i]
			}
		}

		log.Printf("dx: %#v", dx)
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

func (e StdEng) softMaxLastDimF32(output Tensor, x Tensor, axis int, logSoftMax bool) {
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
			z := xArr[i] - maxInput
			exp := math32.Exp(z)

			if logSoftMax {
				outputArr[i] = z
			} else {
				outputArr[i] = exp
			}

			sumExp += exp
		}

		if !logSoftMax {
			sumExp = 1 / sumExp
		}

		for j := 0; j < dimSize; j++ {
			i := ii*dimSize + j

			if logSoftMax {
				outputArr[i] -= math32.Log(sumExp)
			} else {
				outputArr[i] *= sumExp
			}
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
