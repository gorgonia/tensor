package main

import (
	"fmt"
	"io"
)

func writePkgName(f io.Writer, pkg string) {
	switch pkg {
	case tensorPkgLoc:
		fmt.Fprintf(f, "package tensor\n\n // %s\n\n", genmsg)
	case nativePkgLoc:
		fmt.Fprintf(f, "package native\n\n // %s\n\n", genmsg)
	case nativePkgLoc + "_unsafe":
		fmt.Fprintf(f, "// +build !purego \n\npackage native\n\n // %s\n\n", genmsg)
	case nativePkgLoc + "_purego":
		fmt.Fprintf(f, "// +build purego \n\npackage native\n\n // %s\n\n", genmsg)
	case execLoc:
		fmt.Fprintf(f, "package execution\n\n // %s\n\n", genmsg)
	case storageLoc:
		fmt.Fprintf(f, "package storage\n\n // %s\n\n", genmsg)
	default:
		fmt.Fprintf(f, "package unknown\n\n %s\n\n", genmsg)
	}
}

const importUnqualifiedTensor = `import . "gorgonia.org/tensor"
`

const importInternalNative = `import inative "gorgonia.org/tensor/internal/native"
`

const importUnsafe = `import _ "unsafe"
`
