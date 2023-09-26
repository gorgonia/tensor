//go:build !nounsafe
// +build !nounsafe

package tensor

import "github.com/chewxy/inigo/values/tensor/internal"

func MakeDataOrder(fs ...DataOrder) DataOrder { return internal.MakeDataOrder(fs...) }
