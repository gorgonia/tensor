package tensor

import (
	"bytes"
	"encoding/gob"

	"gorgonia.org/tensor/internal/serialization/pb"
	"gorgonia.org/shapes"
)

func (ap AP) GobEncode() (p []byte, err error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	if err = encoder.Encode(ap.shape); err != nil {
		return
	}

	if err = encoder.Encode(ap.strides); err != nil {
		return
	}

	if err = encoder.Encode(ap.o); err != nil {
		return
	}

	if err = encoder.Encode(ap.Δ); err != nil {
		return
	}
	return buf.Bytes(), nil
}

func (ap *AP) GobDecode(p []byte) (err error) {
	buf := bytes.NewBuffer(p)
	decoder := gob.NewDecoder(buf)

	var shape shapes.Shape
	if err = decoder.Decode(&shape); err != nil {
		return
	}

	var strides []int
	if err = decoder.Decode(&strides); err != nil {
		return
	}

	var o DataOrder
	var t Triangle

	if err = decoder.Decode(&o); err != nil {
		return
	}
	if err = decoder.Decode(&t); err != nil {
		return
	}
	ap.Init(shape, strides)
	ap.o = o
	ap.Δ = t
	return nil
}

func (ap *AP) ToPB() *pb.AP {
	pbAP := new(pb.AP)
	pbAP.Shape = make([]int32, len(ap.shape))
	for i := range pbAP.Shape {
		pbAP.Shape[i] = int32(ap.shape[i])
	}

	pbAP.Strides = make([]int32, len(ap.strides))
	for i := range pbAP.Strides {
		pbAP.Strides[i] = int32(ap.strides[i])
	}

	pbAP.O = uint32(ap.o)
	pbAP.T = pb.Triangle(ap.Δ)
	return pbAP
}
