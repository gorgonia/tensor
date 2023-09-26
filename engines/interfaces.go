package stdeng

type eqer1[T any] interface {
	Eq(T) bool
}

type eqer2 interface {
	Eq(any) bool
}
