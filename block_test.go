package weights

import "testing"

// A 2 × 40 tensor: each row has two blocks (32 + 8 values), so 4 scales in total.
func blockTensor() TensorInput {
	data := make([]byte, 80)
	for i := range data {
		data[i] = byte(int8(i%7 - 3))
	}
	return TensorInput{Name: "w", DType: Int8Block32, Shape: []int{2, 40}, Data: data, Scales: []float32{0.5, 2, 1, 4}}
}

func TestInt8Block32_RoundTripAndDequant(t *testing.T) {
	b, err := WriteArtifact("t", 1, TokenizerConfig{}, []TensorInput{blockTensor()})
	if err != nil {
		t.Fatal(err)
	}
	a, err := Open(b)
	if err != nil {
		t.Fatal(err)
	}
	w, _ := a.Tensor("w")
	if w.DType != Int8Block32 {
		t.Fatalf("dtype = %q", w.DType)
	}
	dst := make([]float32, 40)
	if err := w.DequantRow(dst, 1); err != nil {
		t.Fatal(err)
	}
	// row 1 value 0 is data[40] = int8(40%7-3) = 2, block 0 of row 1 has scale 1
	if dst[0] != 2*1 {
		t.Errorf("row 1 col 0 = %v, want 2", dst[0])
	}
	// row 1 col 35 is data[75] = int8(75%7-3) = 2, block 1 of row 1 has scale 4
	if dst[35] != 2*4 {
		t.Errorf("row 1 col 35 = %v, want 8", dst[35])
	}
}

func TestInt8Block32_WrongScaleCountRejected(t *testing.T) {
	in := blockTensor()
	in.Scales = in.Scales[:3]
	b, err := WriteArtifact("t", 1, TokenizerConfig{}, []TensorInput{in})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Open(b); err != ErrScalesMismatch {
		t.Fatalf("Open error = %v, want ErrScalesMismatch", err)
	}
}

func TestDequantRow_Int8AndFloat32AndBounds(t *testing.T) {
	i8 := Tensor{DType: Int8, Shape: []int{1, 3}, Data: []byte{1, 0xFF, 2}, Scales: []float32{0.5}}
	dst := make([]float32, 3)
	if err := i8.DequantRow(dst, 0); err != nil || dst[0] != 0.5 || dst[1] != -0.5 || dst[2] != 1 {
		t.Fatalf("int8 row = %v, %v", dst, err)
	}
	if err := i8.DequantRow(dst, 1); err != ErrRowOutOfRange {
		t.Fatalf("row 1 of 1 = %v, want ErrRowOutOfRange", err)
	}
	if err := i8.DequantRow(dst[:2], 0); err != ErrDstTooShort {
		t.Fatalf("short dst = %v, want ErrDstTooShort", err)
	}
}
