package tensor

import "testing"

func TestMemoryFlag(t *testing.T) {
	var defaultFlag MemoryFlag
	if defaultFlag.manuallyManaged() || !defaultFlag.nativelyAccessible() {
		t.Errorf("Something went wrong with the creation of flags")
	}

	a := ManuallyManaged
	if !a.manuallyManaged() {
		t.Errorf("Expected ManuallyManaged to be true")
	}
	if !a.nativelyAccessible() {
		t.Errorf("Expected ManuallyManaged to be nativelyAccessible")
	}

	b := NativelyInaccessible
	if b.manuallyManaged() {
		t.Errorf("Expected NativelyInaccessible to not be manually managed")
	}
	if b.nativelyAccessible() {
		t.Errorf("Expected NativelyInaccessible to be false %v", b.nativelyAccessible())
	}

	c := MakeMemoryFlag(ManuallyManaged, NativelyInaccessible)
	if !c.manuallyManaged() {
		t.Errorf("Expected c to be manually managed")
	}
	if c.nativelyAccessible() {
		t.Errorf("Expected c to be natively inaccessible")
	}
}

func TestDataOrder(t *testing.T) {
	var defaultFlag DataOrder
	if defaultFlag.IsColMajor() || defaultFlag.IsNotContiguous() || defaultFlag.IsTransposed() {
		t.Error("Expected default flag to be row major and contiguous and not transposed")
	}
	if !(defaultFlag.IsRowMajor() && defaultFlag.IsContiguous()) {
		t.Error("Expected default flag to be row major and contiguous")
	}
	if defaultFlag.String() != "Contiguous, RowMajor" {
		t.Errorf("Expected string is \"Contiguous, RowMajor\". Got %q", defaultFlag.String())
	}

	ncrm := MakeDataOrder(NonContiguous)
	if ncrm.IsColMajor() || ncrm.IsContiguous() {
		t.Error("Expected noncontiguous row major.")
	}
	if ncrm.String() != "NonContiguous, RowMajor" {
		t.Errorf("Expected string is \"NonContiguous, RowMajor\". Got %q", defaultFlag.String())
	}

	cm := ColMajor
	if cm.IsRowMajor() {
		t.Error("colMajor cannot be rowMajor")
	}
	if cm.IsNotContiguous() {
		t.Error("ColMajor by default is contiguous")
	}
	if cm.String() != "Contiguous, ColMajor" {
		t.Errorf(`Expected string is "Contiguous, ColMajor". Got %q`, cm.String())
	}

	// check toggle
	rm := cm.toggleColMajor()
	if rm.IsColMajor() {
		t.Errorf("toggled cm should be rm")
	}

	cm = rm.toggleColMajor()
	if cm.IsRowMajor() {
		t.Errorf("toggled rm should be cm")
	}

	transposed := MakeDataOrder(Transposed)
	if !transposed.IsTransposed() {
		t.Error("Expected transposed flag to be set")
	}
	if transposed.String() != "Contiguous, RowMajorᵀ" {
		t.Errorf("Expected string is \"Contiguous, RowMajorᵀ\". Got %q", defaultFlag.String())
	}
	untransposed := transposed.clearTransposed()
	if untransposed != defaultFlag {
		t.Error("Expected default flag after untransposing")
	}

}
