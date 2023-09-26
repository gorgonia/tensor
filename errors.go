package tensor

type noopError struct{}

func (e noopError) Error() string { return "NO-OP" }
func (e noopError) NoOp()         {}

// NoOpError is a useful for operations that have no op.
type NoOpError interface {
	error
	NoOp()
}

func handleNoOp(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := err.(NoOpError); ok {
		return nil
	}
	return err
}
