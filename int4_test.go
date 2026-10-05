package weights

import (
	"math"
	"testing"
)

func TestInt4Block32_QuantizeKnownBlock(t *testing.T) {
	row := make([]float32, 32)
	row[0] = -2    // largest magnitude, negative: scale = -2 / -8 = 0.25, code 0
	row[16] = 1.75 // 1.75 / 0.25 = 7 → code 15, high nibble of byte 0
	row[1] = 0     // code 8
	q, s, err := QuantizeInt4Block32(row)
	if err != nil {
		t.Fatal(err)
	}
	if len(q) != 16 || len(s) != 1 || s[0] != 0.25 {
		t.Fatalf("q %d bytes, scales %v", len(q), s)
	}
	if q[0] != 0xF0 {
		t.Errorf("byte 0 = %#x, want 0xF0 (value 0 low = 0, value 16 high = 15)", q[0])
	}
	if q[1] != 0x88 {
		t.Errorf("byte 1 = %#x, want 0x88 (zeros)", q[1])
	}
}

func TestInt4Block32_RoundTripError(t *testing.T) {
	const rows, cols = 4, 64
	var data []byte
	var scales []float32
	orig := make([]float32, rows*cols)
	for i := range orig {
		orig[i] = float32(math.Sin(float64(i)*0.37)) * float32(1+i%5)
	}
	for r := 0; r < rows; r++ {
		q, s, err := QuantizeInt4Block32(orig[r*cols : (r+1)*cols])
		if err != nil {
			t.Fatal(err)
		}
		data, scales = append(data, q...), append(scales, s...)
	}
	b, err := WriteArtifact("t", 1, TokenizerConfig{}, []TensorInput{{Name: "w", DType: Int4Block32, Shape: []int{rows, cols}, Data: data, Scales: scales}})
	if err != nil {
		t.Fatal(err)
	}
	a, err := Open(b)
	if err != nil {
		t.Fatal(err)
	}
	w, _ := a.Tensor("w")
	if w.DType != Int4Block32 {
		t.Fatalf("dtype = %q", w.DType)
	}
	dst := make([]float32, cols)
	for r := 0; r < rows; r++ {
		if err := w.DequantRow(dst, r); err != nil {
			t.Fatal(err)
		}
		for c, v := range dst {
			s := scales[r*cols/32+c/32]
			if s < 0 {
				s = -s
			}
			// Q4_0 is asymmetric: codes reach −8 steps on the side of the block's largest value
			// and +7 on the other, so a value beyond 7 steps there is clipped (as in llama.cpp).
			bound := float64(s / 2)
			if over := math.Abs(float64(orig[r*cols+c])) - 7*float64(s); over > bound {
				bound = over
			}
			if d := math.Abs(float64(v - orig[r*cols+c])); d > bound+1e-6 {
				t.Fatalf("row %d col %d: %v decoded as %v (step %v)", r, c, orig[r*cols+c], v, s)
			}
		}
	}
}

func TestInt4Block32_ZeroBlock(t *testing.T) {
	q, s, err := QuantizeInt4Block32(make([]float32, 32))
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range q {
		if b != 0x88 {
			t.Fatalf("byte %#x, want 0x88", b)
		}
	}
	dst := make([]float32, 32)
	DequantInt4Block32(dst, q, s)
	for _, v := range dst {
		if v != 0 {
			t.Fatalf("decoded %v, want 0", v)
		}
	}
}

func TestInt4Block32_RejectsCols(t *testing.T) {
	if _, _, err := QuantizeInt4Block32(make([]float32, 40)); err != ErrInt4Cols {
		t.Fatalf("quantize 40 values: %v", err)
	}
	b, err := WriteArtifact("t", 1, TokenizerConfig{}, []TensorInput{{Name: "w", DType: Int4Block32, Shape: []int{2, 40}, Data: make([]byte, 40), Scales: make([]float32, 4)}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Open(b); err != ErrInt4Cols {
		t.Fatalf("Open = %v, want ErrInt4Cols", err)
	}
}

func TestInt4Block32_WrongScaleCount(t *testing.T) {
	b, err := WriteArtifact("t", 1, TokenizerConfig{}, []TensorInput{{Name: "w", DType: Int4Block32, Shape: []int{2, 32}, Data: make([]byte, 32), Scales: make([]float32, 1)}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Open(b); err != ErrScalesMismatch {
		t.Fatalf("Open = %v, want ErrScalesMismatch", err)
	}
}

func TestInt4Block32_DTypeByte(t *testing.T) {
	if dtypeToByte(Int4Block32) != 6 {
		t.Fatalf("Int4Block32 byte = %d, want 6", dtypeToByte(Int4Block32))
	}
	if d, err := byteToDType(3); err != nil || d != Int4 {
		t.Fatalf("byte 3 = %q, %v; the old Int4 must still open", d, err)
	}
}
