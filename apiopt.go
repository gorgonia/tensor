package tensor

import (
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

type APIOptions struct{}
type APIOpt func(*APIOptions)

func postOpBroadcastReshape(behav internal.BroadcastBehaviour, t, u, retVal DescWithStorage) (err error) {
	broadcastDir := behav.BroadcastShape()
	switch broadcastDir {
	case internal.BroadcastShapeLeft:
		err = retVal.Reshape(u.Shape()...)
	case internal.BroadcastShapeRight:
		err = retVal.Reshape(t.Shape()...)
	case internal.BroadcastShapeLeft | internal.BroadcastShapeRight:
		err = errors.Errorf(errors.NYIPR, "reshaping for bidirectional broadcasting", "postOpBroadcast")
	}
	return
}
