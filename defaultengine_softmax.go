package tensor

import (
	"context"
	"fmt"
	"math"
	"sync"

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

// SoftMax performs the softmax operation on the given tensor. Currently it expects the tensor to be a Dense tensor.
// Please make a pull request to support sparse tensors.
//
// The softmax function is defined as :
//	σ(x) = e^x_i / Σ(e^x_i)
func (e StdEng) SoftMax(x Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	axis = resolveAxis(axis, x.Dims())
	expectedShape := x.Shape()

	var reuse DenseTensor
	var safe, toReuse, _ bool
	var ctx context.Context
	if ctx, reuse, safe, toReuse, _, _, err = handleFuncOpts(expectedShape, x.Dtype(), x.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if err = handleCtx(ctx); err != nil {
		return nil, err // this err will be noopError{}, no need to wrap.
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

// SoftMaxB computes gradient of the input `x`, given the `output = SoftMax(x)` and its associated gradient. Currently it expects the tensor to be a Dense tensor.
// Please make a pull request to support sparse tensors.
func (e StdEng) SoftMaxB(output, grad Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	if !output.Shape().Eq(grad.Shape()) {
		return nil, fmt.Errorf("output and grad shapes don't match")
	}

	if !output.Dtype().Eq(grad.Dtype()) {
		return nil, fmt.Errorf("output and grad types don't match")
	}

	axis = resolveAxis(axis, output.Dims())
	expectedShape := output.Shape()

	var reuse DenseTensor
	var safe, toReuse, _ bool
	var ctx context.Context
	if ctx, reuse, safe, toReuse, _, _, err = handleFuncOpts(expectedShape, output.Dtype(), output.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if err = handleCtx(ctx); err != nil {
		return nil, err // this err will be noopError{}, no need to wrap.
	}

	if safe || !toReuse && reuse == nil && safe {
		// create reuse
		reuse = New(WithShape(expectedShape...), Of(output.Dtype()))
	}

	switch output.Dtype() {
	case Float32:
		if expectedShape.Dims()-1 == axis {
			e.softMaxBLastDimF32(reuse, output, grad, axis, false)
		} else {
			e.softMaxBInnerDimF32(reuse, output, grad, axis, false)
		}
	case Float64:
		if expectedShape.Dims()-1 == axis {
			e.softMaxBLastDimF64(reuse, output, grad, axis, false)
		} else {
			e.softMaxBInnerDimF64(reuse, output, grad, axis, false)
		}
	default:
		return nil, fmt.Errorf("type %v not supported", output.Dtype())
	}

	return reuse, nil
}

// LogSoftMax performs softmax but in log space. This provides some amount of numerical stabilization.
// Conceptually it is the same as performing a logarithm after applying the softmax function.
// Currently it expects the tensor to be a Dense tensor.
// Please make a pull request to support sparse tensors.
func (e StdEng) LogSoftMax(x Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	axis = resolveAxis(axis, x.Dims())
	expectedShape := x.Shape()

	var reuse DenseTensor
	var safe, toReuse, _ bool
	var ctx context.Context
	if ctx, reuse, safe, toReuse, _, _, err = handleFuncOpts(expectedShape, x.Dtype(), x.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if err = handleCtx(ctx); err != nil {
		return nil, err // this err will be noopError{}, no need to wrap.
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

// LogSoftMaxB computes the gradient of the input `x`, given the `output = LogSoftmax(x)` and its associated gradient.
// Currently it expects the tensor to be a Dense tensor.
// Please make a pull request to support sparse tensors.
func (e StdEng) LogSoftMaxB(output, grad Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	if !output.Shape().Eq(grad.Shape()) {
		return nil, fmt.Errorf("output and grad shapes don't match")
	}

	if !output.Dtype().Eq(grad.Dtype()) {
		return nil, fmt.Errorf("output and grad types don't match")
	}

	axis = resolveAxis(axis, output.Dims())
	expectedShape := output.Shape()

	var reuse DenseTensor
	var safe, toReuse, _ bool
	var ctx context.Context
	if ctx, reuse, safe, toReuse, _, _, err = handleFuncOpts(expectedShape, output.Dtype(), output.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if err = handleCtx(ctx); err != nil {
		return nil, err // this err will be noopError{}, no need to wrap.
	}

	if safe || !toReuse && reuse == nil && safe {
		// create reuse
		reuse = New(WithShape(expectedShape...), Of(output.Dtype()))
	}

	switch output.Dtype() {
	case Float32:
		if expectedShape.Dims()-1 == axis {
			e.softMaxBLastDimF32(reuse, output, grad, axis, true)
		} else {
			e.softMaxBInnerDimF32(reuse, output, grad, axis, true)
		}
	case Float64:
		if expectedShape.Dims()-1 == axis {
			e.softMaxBLastDimF64(reuse, output, grad, axis, true)
		} else {
			e.softMaxBInnerDimF64(reuse, output, grad, axis, true)
		}
	default:
		return nil, fmt.Errorf("type %v not supported", output.Dtype())
	}

	return reuse, nil
}

func (e StdEng) softMaxLastDimF64(output Tensor, x Tensor, axis int, logSoftMax bool) {
	outputArr := getFloat64s(output)
	xArr := getFloat64s(x)

	xShape := x.Shape()

	outerSize := 1
	dimSize := xShape[axis]
	for i := 0; i < axis; i++ {
		outerSize *= xShape[i]
	}

	var wg sync.WaitGroup
	for ii := 0; ii < outerSize; ii++ {
		wg.Add(1)
		go func(ii int, wg *sync.WaitGroup) {
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
			wg.Done()
		}(ii, &wg)

	}
	wg.Wait()
}

func (e StdEng) softMaxBLastDimF64(inputGrad, output, grad Tensor, axis int, logSoftMax bool) {
	dx := getFloat64s(inputGrad)
	outputArr := getFloat64s(output)
	gradArr := getFloat64s(grad)

	outputShape := output.Shape()

	outerSize := 1
	dimSize := outputShape[axis]
	for i := 0; i < axis; i++ {
		outerSize *= outputShape[i]
	}

	var wg sync.WaitGroup
	for ii := 0; ii < outerSize; ii++ {
		wg.Add(1)
		if logSoftMax {
			go func(gradArr, dx []float64, ii int, wg *sync.WaitGroup) {
				sum := gradArr[ii*dimSize]
				for j := 1; j < dimSize; j++ {
					i := ii*dimSize + j

					sum += gradArr[i]
				}

				for j := 0; j < dimSize; j++ {
					i := ii*dimSize + j

					dx[i] = gradArr[i] - (math.Exp(outputArr[i]) * sum)
				}
				wg.Done()
			}(gradArr, dx, ii, &wg)

		} else {
			go func(outputArr, gradArr, dx []float64, ii int, wg *sync.WaitGroup) {
				//mul := make([]float64, dimSize)
				var sum float64
				for j := 0; j < dimSize; j++ {
					i := ii*dimSize + j

					//mul[j] = outputArr[i] * gradArr[i]
					sum += outputArr[i] * gradArr[i]
				}

				// sum := mul[0]
				// for j := 1; j < dimSize; j++ {
				// 	sum += mul[j]
				// }

				for j := 0; j < dimSize; j++ {
					i := ii*dimSize + j
					dx[i] = (gradArr[i] - sum) * outputArr[i]
				}
				wg.Done()
			}(outputArr, gradArr, dx, ii, &wg)
		}
	}
	wg.Wait()
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

	outputArr := getFloat64s(output)
	xArr := getFloat64s(x)

	var wg sync.WaitGroup
	for ii := 0; ii < innerSize*outerSize; ii++ {
		wg.Add(1)
		go func(ii int, wg *sync.WaitGroup) {
			outerIndex, innerIndex := divmod(ii, innerSize)

			inputPart := xArr[outerIndex*outerStride+innerIndex:]
			outputPart := outputArr[outerIndex*outerStride+innerIndex:]

			maxInput := inputPart[0]
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
			wg.Done()
		}(ii, &wg)
	}
	wg.Wait()
}

func (e StdEng) softMaxBInnerDimF64(inputGrad, output, grad Tensor, axis int, logSoftmax bool) {
	dxShape := inputGrad.Shape()

	innerSize, outerSize := 1, 1
	for i := 0; i < axis; i++ {
		outerSize *= dxShape[i]
	}

	for i := axis + 1; i < dxShape.Dims(); i++ {
		innerSize *= dxShape[i]
	}

	dimSize := dxShape[axis]
	dimStride := innerSize
	outerStride := dimSize * dimStride

	dxArr := getFloat64s(inputGrad)
	outputArr := getFloat64s(output)
	gradArr := getFloat64s(grad)

	var wg sync.WaitGroup
	for ii := 0; ii < innerSize*outerSize; ii++ {
		wg.Add(1)
		go func(ii int, wg *sync.WaitGroup) {
			outerIndex, innerIndex := divmod(ii, innerSize)

			gradPart := gradArr[outerIndex*outerStride+innerIndex:]
			dxPart := dxArr[outerIndex*outerStride+innerIndex:]
			outputPart := outputArr[outerIndex*outerStride+innerIndex:]

			sum := 0.0
			for j := 0; j < dimSize; j++ {
				i := j * dimStride

				if logSoftmax {
					sum += gradPart[i]
				} else {
					sum += gradPart[i] * outputPart[i]
				}
			}

			for j := 0; j < dimSize; j++ {
				i := j * dimStride

				if logSoftmax {
					dxPart[i] = gradPart[i] - math.Exp(outputPart[i])*sum
				} else {
					dxPart[i] = outputPart[i] * (gradPart[i] - sum)
				}
			}
			wg.Done()
		}(ii, &wg)

	}
	wg.Wait()
}

func (e StdEng) softMaxLastDimF32(output Tensor, x Tensor, axis int, logSoftMax bool) {
	outputArr := getFloat32s(output)
	xArr := getFloat32s(x)
	xShape := x.Shape()

	outerSize := 1
	dimSize := xShape[axis]
	for i := 0; i < axis; i++ {
		outerSize *= xShape[i]
	}

	var wg sync.WaitGroup
	for ii := 0; ii < outerSize; ii++ {
		wg.Add(1)
		go func(ii int, wg *sync.WaitGroup) {
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
			wg.Done()
		}(ii, &wg)
	}
	wg.Wait()
}

func (e StdEng) softMaxBLastDimF32(inputGrad, output, grad Tensor, axis int, logSoftMax bool) {
	dx := getFloat32s(inputGrad)
	outputArr := getFloat32s(output)
	gradArr := getFloat32s(grad)

	outputShape := output.Shape()

	outerSize := 1
	dimSize := outputShape[axis]
	for i := 0; i < axis; i++ {
		outerSize *= outputShape[i]
	}

	var wg sync.WaitGroup
	for ii := 0; ii < outerSize; ii++ {
		wg.Add(1)

		if logSoftMax {
			go func(ii int, wg *sync.WaitGroup) {
				sum := gradArr[ii*dimSize]
				for j := 1; j < dimSize; j++ {
					i := ii*dimSize + j

					sum += gradArr[i]
				}

				for j := 0; j < dimSize; j++ {
					i := ii*dimSize + j

					dx[i] = gradArr[i] - (math32.Exp(outputArr[i]) * sum)
				}
				wg.Done()
			}(ii, &wg)
		} else {
			go func(ii int, wg *sync.WaitGroup) {
				//mul := make([]float32, dimSize)
				var sum float32
				for j := 0; j < dimSize; j++ {
					i := ii*dimSize + j

					//mul[j] = outputArr[i] * gradArr[i]
					sum += outputArr[i] * gradArr[i]
				}

				// sum := mul[0]
				// for j := 1; j < dimSize; j++ {
				// 	sum += mul[j]
				// }

				for j := 0; j < dimSize; j++ {
					i := ii*dimSize + j

					dx[i] = (gradArr[i] - sum) * outputArr[i]
				}
				wg.Done()
			}(ii, &wg)
		}
	}
	wg.Wait()
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

	outputArr := getFloat32s(output)
	xArr := getFloat32s(x)

	var wg sync.WaitGroup
	for ii := 0; ii < innerSize*outerSize; ii++ {
		wg.Add(1)

		go func(ii int, wg *sync.WaitGroup) {
			outerIndex, innerIndex := divmod(ii, innerSize)

			inputPart := xArr[outerIndex*outerStride+innerIndex:]
			outputPart := outputArr[outerIndex*outerStride+innerIndex:]

			maxInput := inputPart[0]
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
			wg.Done()
		}(ii, &wg)
	}
	wg.Wait()
}

func (e StdEng) softMaxBInnerDimF32(inputGrad, output, grad Tensor, axis int, logSoftmax bool) {
	dxShape := inputGrad.Shape()

	innerSize, outerSize := 1, 1
	for i := 0; i < axis; i++ {
		outerSize *= dxShape[i]
	}

	for i := axis + 1; i < dxShape.Dims(); i++ {
		innerSize *= dxShape[i]
	}

	dimSize := dxShape[axis]
	dimStride := innerSize
	outerStride := dimSize * dimStride

	dxArr := getFloat32s(inputGrad)
	outputArr := getFloat32s(output)
	gradArr := getFloat32s(grad)

	var wg sync.WaitGroup
	for ii := 0; ii < innerSize*outerSize; ii++ {
		wg.Add(1)

		go func(ii int, wg *sync.WaitGroup) {
			outerIndex, innerIndex := divmod(ii, innerSize)

			gradPart := gradArr[outerIndex*outerStride+innerIndex:]
			dxPart := dxArr[outerIndex*outerStride+innerIndex:]
			outputPart := outputArr[outerIndex*outerStride+innerIndex:]

			sum := float32(0.0)
			for j := 0; j < dimSize; j++ {
				i := j * dimStride

				if logSoftmax {
					sum += gradPart[i]
				} else {
					sum += gradPart[i] * outputPart[i]
				}
			}

			for j := 0; j < dimSize; j++ {
				i := j * dimStride

				if logSoftmax {
					dxPart[i] = gradPart[i] - math32.Exp(outputPart[i])*sum
				} else {
					dxPart[i] = outputPart[i] * (gradPart[i] - sum)
				}
			}
			wg.Done()
		}(ii, &wg)
	}
	wg.Wait()
}
