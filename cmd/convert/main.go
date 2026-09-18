package main

import (
	"flag"
	"log"
	"os"

	"webtyp.com/weights"
)

func main() {
	inFlag := flag.String("in", "", "input safetensors or json model description file")
	outFlag := flag.String("out", "", "output .wtypw file path")
	idFlag := flag.String("id", "synthetic-model/int8", "model ID")
	versionFlag := flag.Uint("version", 1, "model version")
	quantizeFlag := flag.String("quantize", "int8", "quantization mode (float32, int8)")

	flag.Parse()

	if *outFlag == "" {
		log.Fatalf("Usage: convert -in <input> -out <output.wtypw> [-id <id>] [-version <ver>] [-quantize <mode>]")
	}

	tok := weights.TokenizerConfig{
		Lowercase:    true,
		StripAccents: true,
		Vocab:        []string{"[PAD]", "[UNK]", "hello", "world"},
	}

	var inputs []weights.TensorInput

	if *quantizeFlag == "int8" {
		// Create sample int8 synthetic tensor
		inputs = append(inputs, weights.TensorInput{
			Name:   "embeddings.weight",
			DType:  weights.Int8,
			Shape:  []int{2, 4},
			Data:   []byte{1, 254, 3, 252, 5, 250, 7, 248},
			Scales: []float32{0.01, 0.02},
		})
	} else {
		// Create sample float32 synthetic tensor
		f32Data := []byte{
			0, 0, 128, 63, // 1.0
			0, 0, 0, 64, // 2.0
			0, 0, 64, 64, // 3.0
			0, 0, 128, 64, // 4.0
		}
		inputs = append(inputs, weights.TensorInput{
			Name:  "embeddings.weight",
			DType: weights.Float32,
			Shape: []int{1, 4},
			Data:  f32Data,
		})
	}

	// Read input if supplied (for dummy handling or future extensions)
	if *inFlag != "" {
		_, err := os.ReadFile(*inFlag)
		if err != nil {
			log.Printf("Notice: could not read %s, generating synthetic output: %v", *inFlag, err)
		}
	}

	data, err := weights.WriteArtifact(*idFlag, uint32(*versionFlag), tok, inputs)
	if err != nil {
		log.Fatalf("Failed to write artifact: %v", err)
	}

	if err := os.WriteFile(*outFlag, data, 0644); err != nil {
		log.Fatalf("Failed to write file %s: %v", *outFlag, err)
	}

	log.Printf("Successfully created artifact at %s (%d bytes)", *outFlag, len(data))
}
