package weights_test

import (
	"bytes"
	"errors"
	"reflect"
	"testing"
	"unsafe"

	"webtyp.com/weights"
)

func createTestArtifactData(t *testing.T, id string, ver uint32) []byte {
	t.Helper()
	tok := weights.TokenizerConfig{
		Lowercase:    true,
		StripAccents: true,
		Vocab:        []string{"a", "b", "c"},
	}

	inputs := []weights.TensorInput{
		{
			Name:  "f32_tensor",
			DType: weights.Float32,
			Shape: []int{2, 2},
			Data: []byte{
				0, 0, 128, 63, // 1.0
				0, 0, 0, 64, // 2.0
				0, 0, 64, 64, // 3.0
				0, 0, 128, 64, // 4.0
			},
		},
		{
			Name:   "int8_tensor",
			DType:  weights.Int8,
			Shape:  []int{2, 3},
			Data:   []byte{10, 20, 30, 40, 50, 60},
			Scales: []float32{0.1, 0.2},
		},
	}

	data, err := weights.WriteArtifact(id, ver, tok, inputs)
	if err != nil {
		t.Fatalf("WriteArtifact failed: %v", err)
	}
	return data
}

func TestOpen_RoundTrip(t *testing.T) {
	data := createTestArtifactData(t, "test-model", 1)
	art, err := weights.Open(data)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	if art.ID != "test-model" {
		t.Errorf("got ID %q, want %q", art.ID, "test-model")
	}
	if art.Version != 1 {
		t.Errorf("got Version %d, want 1", art.Version)
	}
	if len(art.Tensors) != 2 {
		t.Fatalf("got %d tensors, want 2", len(art.Tensors))
	}
	if !art.Tokenizer.Lowercase || !art.Tokenizer.StripAccents {
		t.Errorf("tokenizer config mismatch: %+v", art.Tokenizer)
	}
}

func TestOpen_BadMagic(t *testing.T) {
	data := createTestArtifactData(t, "test-model", 1)
	copy(data[0:8], "BADMAGIC")
	_, err := weights.Open(data)
	if !errors.Is(err, weights.ErrBadMagic) {
		t.Errorf("got error %v, want %v", err, weights.ErrBadMagic)
	}
}

func TestOpen_TruncatedTensorData(t *testing.T) {
	data := createTestArtifactData(t, "test-model", 1)
	truncated := data[:len(data)-5]
	_, err := weights.Open(truncated)
	if !errors.Is(err, weights.ErrTruncatedTensorData) {
		t.Errorf("got error %v, want %v", err, weights.ErrTruncatedTensorData)
	}
}

func TestOpen_ChecksumMismatch(t *testing.T) {
	data := createTestArtifactData(t, "test-model", 1)
	data[len(data)-1] ^= 0xFF
	_, err := weights.Open(data)
	if !errors.Is(err, weights.ErrChecksumMismatch) {
		t.Errorf("got error %v, want %v", err, weights.ErrChecksumMismatch)
	}
}

func TestOpen_Alignment(t *testing.T) {
	data := createTestArtifactData(t, "test-model", 1)
	art, err := weights.Open(data)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	for _, ts := range art.Tensors {
		if len(ts.Data) > 0 {
			ptr := uintptr(unsafe.Pointer(&ts.Data[0]))
			if ptr%64 != 0 {
				t.Errorf("tensor %s data pointer 0x%x not 64-byte aligned", ts.Name, ptr)
			}
		}
	}
}

func TestTensor_Float32sZeroCopy(t *testing.T) {
	data := createTestArtifactData(t, "test-model", 1)
	art, err := weights.Open(data)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	ts, ok := art.Tensor("f32_tensor")
	if !ok {
		t.Fatalf("Tensor f32_tensor not found")
	}

	f32s, err := ts.Float32s()
	if err != nil {
		t.Fatalf("Float32s failed: %v", err)
	}

	expected := []float32{1.0, 2.0, 3.0, 4.0}
	if !reflect.DeepEqual(f32s, expected) {
		t.Errorf("got float32s %v, want %v", f32s, expected)
	}

	if unsafe.Pointer(&f32s[0]) != unsafe.Pointer(&ts.Data[0]) {
		t.Errorf("Float32s slice does not alias tensor Data buffer")
	}
}

func TestArtifact_TensorByName(t *testing.T) {
	data := createTestArtifactData(t, "test-model", 1)
	art, err := weights.Open(data)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	ts, ok := art.Tensor("int8_tensor")
	if !ok {
		t.Fatalf("Tensor int8_tensor not found")
	}
	if ts.Name != "int8_tensor" {
		t.Errorf("got tensor name %s, want int8_tensor", ts.Name)
	}

	_, ok = art.Tensor("non_existent")
	if ok {
		t.Errorf("expected false for non_existent tensor")
	}
}

func TestTensor_Row(t *testing.T) {
	data := createTestArtifactData(t, "test-model", 1)
	art, err := weights.Open(data)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	ts, ok := art.Tensor("int8_tensor")
	if !ok {
		t.Fatalf("Tensor int8_tensor not found")
	}

	r0 := ts.Row(0)
	expectedR0 := []byte{10, 20, 30}
	if !bytes.Equal(r0, expectedR0) {
		t.Errorf("row 0 got %v, want %v", r0, expectedR0)
	}

	r1 := ts.Row(1)
	expectedR1 := []byte{40, 50, 60}
	if !bytes.Equal(r1, expectedR1) {
		t.Errorf("row 1 got %v, want %v", r1, expectedR1)
	}

	outOfBounds := ts.Row(2)
	if outOfBounds != nil {
		t.Errorf("expected nil for out of bounds row index")
	}
}

