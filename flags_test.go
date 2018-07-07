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
	if defaultFlag.IsColMajor() || defaultFlag.IsNotContiguous() {
		t.Error("Expected default flag to be row major and contiguous")
	}
	if !(defaultFlag.IsRowMajor() && defaultFlag.IsContiguous()) {
		t.Error("Expected default flag to be row major and contiguous")
	}
	if defaultFlag.String() != "Contiguous, RowMajor" {
		t.Error("Expected string is \"Contiguous, RowMajor\"")
	}

	cm := ColMajor
	if cm.IsRowMajor() {
		t.Error("colMajor cannot be rowMajor")
	}
	if cm.IsNotContiguous() {
		t.Error("ColMajor by default is contiguous")
	}
	if cm.String() != "Contiguous, ColMajor" {
		t.Error(`Expected string is "Contiguous, ColMajor"`)
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
}
