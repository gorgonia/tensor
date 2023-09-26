package internal

var intPool []int = make([]int, 0, 1024)

func BorrowInts(n int) []int {
	if n <= 0 {
		return nil
	}
	if n <= len(intPool) {
		x := intPool[:n:n]
		intPool = intPool[n:]
		return x
	}
	return make([]int, n)
}

func ReturnInts(is []int) {}
