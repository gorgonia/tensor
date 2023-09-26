//go:build !nounsafe
// +build !nounsafe

package tensor

import "gorgonia.org/tensor/internal"

func MakeDataOrder(fs ...DataOrder) DataOrder { return internal.MakeDataOrder(fs...) }
