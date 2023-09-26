package tensor

// Dear John C Reynolds, you are an inspiration. I am so sorry for this bunch of code that butchers the concepts of defunctionalization.

type TensorOf[T, DT any] interface {
	TensorType(T)
	DataType(DT)
}
