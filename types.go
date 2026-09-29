package weights

import (
	"unsafe"
)

// Error represents package error constants.
type Error string

func (e Error) Error() string { return string(e) }

const (
	ErrBadMagic            = Error("weights: bad magic")
	ErrBadVersion          = Error("weights: unsupported version")
	ErrBadAlignment        = Error("weights: tensor data not aligned")
	ErrChecksumMismatch    = Error("weights: checksum mismatch")
	ErrTruncatedTensorData = Error("weights: truncated tensor data")
	ErrInvalidDType        = Error("weights: invalid dtype")
	ErrInvalidHeader       = Error("weights: invalid header")
	ErrScalesMismatch      = Error("weights: scale count does not match the tensor shape")
	ErrRowOutOfRange       = Error("weights: row out of range")
	ErrDstTooShort         = Error("weights: destination shorter than a row")
)

// DType represents tensor data types.
type DType string

const (
	Float32 DType = "float32"
	Int8    DType = "int8"
	Uint8   DType = "uint8"
	Int4    DType = "int4"
	Float16 DType = "float16"

	// Int8Block32 is a 2-D tensor of int8 values, row-major, with one float32 scale per block of
	// BlockSize consecutive values of a row (the layout of GGUF Q8_0): value = q × scale.
	// Scales holds rows × ceil(cols / BlockSize) entries, row by row.
	Int8Block32 DType = "int8b32"
)

// BlockSize is the number of consecutive values of a row that share one scale in Int8Block32.
const BlockSize = 32

func dtypeToByte(d DType) byte {
	switch d {
	case Float32:
		return 0
	case Int8:
		return 1
	case Uint8:
		return 2
	case Int4:
		return 3
	case Float16:
		return 4
	case Int8Block32:
		return 5
	default:
		return 255
	}
}

func byteToDType(b byte) (DType, error) {
	switch b {
	case 0:
		return Float32, nil
	case 1:
		return Int8, nil
	case 2:
		return Uint8, nil
	case 3:
		return Int4, nil
	case 4:
		return Float16, nil
	case 5:
		return Int8Block32, nil
	default:
		return "", ErrInvalidDType
	}
}

// TokenizerConfig holds tokenizer properties stored in the artifact header.
type TokenizerConfig struct {
	Lowercase    bool
	StripAccents bool
	Vocab        []string
}

// Tensor represents a single parameter tensor within an artifact.
type Tensor struct {
	Name   string
	DType  DType
	Shape  []int
	Data   []byte
	Scales []float32
}

// Float32s returns a zero-copy float32 slice view into Data when DType is Float32.
// Precondition: Data slice pointer must be 4-byte memory aligned.
func (t Tensor) Float32s() ([]float32, error) {
	if t.DType != Float32 {
		return nil, ErrInvalidDType
	}
	if len(t.Data)%4 != 0 {
		return nil, ErrTruncatedTensorData
	}
	if len(t.Data) == 0 {
		return nil, nil
	}
	ptr := uintptr(unsafe.Pointer(&t.Data[0]))
	if ptr%4 != 0 {
		return nil, ErrBadAlignment
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&t.Data[0])), len(t.Data)/4), nil
}

// Row returns the byte slice corresponding to row i of the tensor.
func (t Tensor) Row(i int) []byte {
	if i < 0 || len(t.Shape) == 0 {
		return nil
	}
	rows := t.Shape[0]
	if rows <= 0 || i >= rows {
		return nil
	}
	rowBytes := len(t.Data) / rows
	if rowBytes <= 0 {
		return nil
	}
	start := i * rowBytes
	end := start + rowBytes
	if start >= len(t.Data) || end > len(t.Data) {
		return nil
	}
	return t.Data[start:end]
}

// Cols is the number of values in one row: the product of every dimension after the first.
func (t Tensor) Cols() int {
	if len(t.Shape) == 0 {
		return 0
	}
	c := 1
	for _, d := range t.Shape[1:] {
		c *= d
	}
	return c
}

// blocksPerRow is ceil(cols / BlockSize).
func blocksPerRow(cols int) int { return (cols + BlockSize - 1) / BlockSize }

// DequantRow writes row i of the tensor into dst as float32, whatever its storage: Float32 is
// copied, Int8 uses the row's scale, Int8Block32 uses one scale per BlockSize values.
func (t Tensor) DequantRow(dst []float32, i int) error {
	if len(t.Shape) == 0 || i < 0 || i >= t.Shape[0] {
		return ErrRowOutOfRange
	}
	cols := t.Cols()
	if len(dst) < cols {
		return ErrDstTooShort
	}
	switch t.DType {
	case Float32:
		f, err := t.Float32s()
		if err != nil {
			return err
		}
		copy(dst[:cols], f[i*cols:(i+1)*cols])
	case Int8:
		if len(t.Scales) != t.Shape[0] {
			return ErrScalesMismatch
		}
		s := t.Scales[i]
		row := t.Data[i*cols : (i+1)*cols]
		for c, b := range row {
			dst[c] = float32(int8(b)) * s
		}
	case Int8Block32:
		nb := blocksPerRow(cols)
		if len(t.Scales) != t.Shape[0]*nb {
			return ErrScalesMismatch
		}
		scales := t.Scales[i*nb : (i+1)*nb]
		row := t.Data[i*cols : (i+1)*cols]
		for c, b := range row {
			dst[c] = float32(int8(b)) * scales[c/BlockSize]
		}
	default:
		return ErrInvalidDType
	}
	return nil
}

// Artifact represents a loaded model artifact containing tensors and configuration.
type Artifact struct {
	ID        string
	Version   uint32
	Tensors   []Tensor
	Tokenizer TokenizerConfig
}

// Tensor returns the tensor stored under name, or false if not found.
func (a *Artifact) Tensor(name string) (Tensor, bool) {
	if a == nil {
		return Tensor{}, false
	}
	for i := range a.Tensors {
		if a.Tensors[i].Name == name {
			return a.Tensors[i], true
		}
	}
	return Tensor{}, false
}
