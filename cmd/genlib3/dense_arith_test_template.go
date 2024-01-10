package main

import "text/template"

const idenTestsRaw = `func gen{{.Name}}Iden[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset({{.Identity}}); err != nil {
			t.Errorf("Memset {{.Identity}} failed: %v", err)
			return false
		}
		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.{{.Name}}(b)
		if err, retEarly := qcErrCheck(t, "{{.Name}}", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func gen{{.Name}}IdenUnsafe[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset({{.Identity}}); err != nil {
			t.Errorf("Memset {{.Identity}} failed: %v", err)
			return false
		}
		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.{{.Name}}(b, tensor.UseUnsafe)
		if err, retEarly := qcErrCheck(t, "{{.Name}} (Unsafe)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.Same(ret, a) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func gen{{.Name}}IdenReuse[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset({{.Identity}}); err != nil {
			t.Errorf("Memset {{.Identity}} failed: %v", err)
			return false
		}
		correct := a.Clone()
		reuse := b.Clone()
		if err := reuse.Memset(1); err != nil {
			return false
		}

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.{{.Name}}(b, tensor.WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "{{.Name}} (Reuse)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.Same(ret, reuse) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func gen{{.Name}}IdenIncr[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset({{.Identity}}); err != nil {
			t.Errorf("Memset {{.Identity}} failed: %v", err)
			return false
		}
		correct := a.Clone()
		incr := b.Clone()
		incr.Zero()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.{{.Name}}(b, tensor.WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "{{.Name}} (Incr)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.Same(ret, incr) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func gen{{.Name}}IdenBroadcast[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a, b *Dense[DT]) bool {
		if err := b.Memset({{.Identity}}); err != nil {
			t.Errorf("Memset {{.Identity}} failed: %v", err)
			return false
		}
		correct := a.Clone()
		correctShape := largestShape(a.Shape(), b.Shape())
		if err := correct.Reshape(correctShape...); err != nil {
			t.Errorf("While reshaping, err: %v", err)
			return false
		}

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}

		// TODO: this check is only because broadcasting doesn't work with tensors that require iterators yet. When
		// it does, this section should be removed.
		if !a.DataOrder().HasSameOrder(b.DataOrder()) && // iterators required
			a.Shape().TotalSize() != b.Shape().TotalSize() && !(a.Shape().IsScalar() && b.Shape().IsScalar()) { // but not fastpath
			we = true
		}

		_, ok := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.{{.Name}}(b, tensor.AutoBroadcast)
		if err, retEarly := qcErrCheck(t, "{{.Name}} (Broadcast)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.True(correct.Shape().Eq(ret.Shape())) &&
			assert.Equal(correct.Data(), ret.Data())
	}
}
`

const invTestsRaw = `
func gen{{.Name}}Inv[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.{{.Name}}(b)
		if err, retEarly := qcErrCheck(t, "{{.Name}}", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.{{.Inverse}}(b, tensor.UseUnsafe)
		return assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func gen{{.Name}}InvUnsafe[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.{{.Name}}(b, tensor.UseUnsafe)
		if err, retEarly := qcErrCheck(t, "{{.Name}} (Unsafe)", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.{{.Inverse}}(b, tensor.UseUnsafe)
		return assert.Same(a, ret) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func gen{{.Name}}InvReuse[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

		correct := a.Clone()
		reuse := b.Clone()
		if err := reuse.Memset(1); err != nil{
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.{{.Name}}(b, tensor.WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "{{.Name}} (Reuse)", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.{{.Inverse}}(b, tensor.UseUnsafe)
		return assert.Same(ret, reuse) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func gen{{.Name}}InvBroadcast[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a, b *Dense[DT]) bool {
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}
		correct := a.Clone()
		correctShape := largestShape(a.Shape(), b.Shape())
		if err := correct.Reshape(correctShape...); err != nil {
			t.Errorf("While reshaping, err: %v", err)
			return false
		}

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}

		// TODO: this check is only because broadcasting doesn't work with tensors that require iterators yet. When
		// it does, this section should be removed.
		if !a.DataOrder().HasSameOrder(b.DataOrder()) && // iterators required
			a.Shape().TotalSize() != b.Shape().TotalSize() && !(a.Shape().IsScalar() && b.Shape().IsScalar()) { // but not fastpath
			we = true
		}

		_, ok := a.Engine().(tensor.{{.Interface}}[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.{{.Name}}(b, tensor.AutoBroadcast)
		if err, retEarly := qcErrCheck(t, "{{.Name}} (Broadcast)", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.{{.Inverse}}(b, tensor.UseUnsafe, tensor.AutoBroadcast)
		return assert.True(correct.Shape().Eq(ret.Shape())) &&
			assert.True(allClose(correct.Data(), ret.Data()), "Expected ret to be close to correct.\nCorrect: %v\nGot: %v", correct.Data(), ret.Data())

	}
}
`

const denseArithMethodTestRaw = `func TestDense_{{.Name}}(t *testing.T) {
	assert := assert.New(t)
	{{$N := .Name}}
{{if ne .Identity "" -}}
	{{- range $id, $dt := .Datatypes -}}
	qcHelper[{{$dt}}](t, assert, gen{{$N}}Iden[{{$dt}}])
	qcHelper[{{$dt}}](t, assert, gen{{$N}}IdenUnsafe[{{$dt}}])
	qcHelper[{{$dt}}](t, assert, gen{{$N}}IdenReuse[{{$dt}}])
	qcHelper[{{$dt}}](t, assert, gen{{$N}}IdenIncr[{{$dt}}])
	qcHelper[{{$dt}}](t, assert, gen{{$N}}IdenBroadcast[{{$dt}}])
	{{ end -}}
{{end}}


{{if ne .Inverse "" -}}
	{{- range $id, $dt := .Datatypes -}}
	qcHelper[{{$dt}}](t, assert, gen{{$N}}Inv[{{$dt}}])
	qcHelper[{{$dt}}](t, assert, gen{{$N}}InvUnsafe[{{$dt}}])
	qcHelper[{{$dt}}](t, assert, gen{{$N}}InvReuse[{{$dt}}])
	qcHelper[{{$dt}}](t, assert, gen{{$N}}InvBroadcast[{{$dt}}])
	{{end}}
{{end}}

}

`

var (
	idenTests            *template.Template
	invTests             *template.Template
	denseArithMethodTest *template.Template
)

func init() {
	idenTests = template.Must(template.New("identityQCFuncs").Funcs(funcs).Parse(idenTestsRaw))
	invTests = template.Must(template.New("invQCFuncs").Funcs(funcs).Parse(invTestsRaw))
	denseArithMethodTest = template.Must(template.New("denseArithMethodTest").Funcs(funcs).Parse(denseArithMethodTestRaw))
}
