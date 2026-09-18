package main

import (
	"os"
	"path/filepath"
	"testing"

	"webtyp.com/weights"
)

func TestConvertCommand(t *testing.T) {
	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "test.wtypw")

	tok := weights.TokenizerConfig{
		Lowercase:    true,
		StripAccents: true,
		Vocab:        []string{"[PAD]", "[UNK]", "hello", "world"},
	}

	inputs := []weights.TensorInput{
		{
			Name:   "embeddings.weight",
			DType:  weights.Int8,
			Shape:  []int{2, 4},
			Data:   []byte{1, 254, 3, 252, 5, 250, 7, 248},
			Scales: []float32{0.01, 0.02},
		},
	}

	data, err := weights.WriteArtifact("synthetic-model/int8", 1, tok, inputs)
	if err != nil {
		t.Fatalf("WriteArtifact failed: %v", err)
	}

	if err := os.WriteFile(outPath, data, 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	readData, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}

	art, err := weights.Open(readData)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	if art.ID != "synthetic-model/int8" {
		t.Errorf("got ID %s, want synthetic-model/int8", art.ID)
	}
	if len(art.Tensors) != 1 {
		t.Errorf("got %d tensors, want 1", len(art.Tensors))
	}
}
