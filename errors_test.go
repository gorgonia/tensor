package tensor

import (
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestHandleNoOp(t *testing.T) {
	otherErr := errors.New("other error")

	assert.Equal(t, nil, handleNoOp(noopError{}))
	assert.Equal(t, nil, handleNoOp(nil))
	assert.Equal(t, otherErr, handleNoOp(otherErr))
}
