package paginated

// IsScalar will always return false for a
// paginated tensor.
func (p *Tensor) IsScalar() bool {
	return false
}

// ScalarValue does not return anything for a paginated tensor.
func (p *Tensor) ScalarValue() interface{} {
	return nil
}
