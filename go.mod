module gorgonia.org/tensor

go 1.13

replace gorgonia.org/dtype => /home/chewxy/workspace/gorgoniaws/src/gorgonia.org/dtype

require (
	github.com/apache/arrow/go/arrow v0.0.0-20201229220542-30ce2eb5d4dc
	github.com/chewxy/math32 v1.0.6
	github.com/gogo/protobuf v1.3.1
	github.com/golang/protobuf v1.4.3
	github.com/google/flatbuffers v1.12.0
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.7.0
	go4.org/unsafe/assume-no-moving-gc v0.0.0-20201222180813-1025295fd063
	gonum.org/v1/gonum v0.8.2
	gorgonia.org/dtype v0.0.0-00010101000000-000000000000
	gorgonia.org/vecf32 v0.9.0
	gorgonia.org/vecf64 v0.9.0
)
