package weights_test

import (
	"context"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"reflect"
	"testing"

	"webtyp.com/weights"
)

type memoryStorage struct {
	m     map[string][]byte
	quota int64
	usage int64
}

func newMemoryStorage(quota int64) *memoryStorage {
	return &memoryStorage{
		m:     make(map[string][]byte),
		quota: quota,
	}
}

func (ms *memoryStorage) Get(key string) ([]byte, error) {
	val, ok := ms.m[key]
	if !ok {
		return nil, errors.New("not found")
	}
	return val, nil
}

func (ms *memoryStorage) Put(key string, val []byte) error {
	oldLen := len(ms.m[key])
	ms.m[key] = val
	ms.usage += int64(len(val) - oldLen)
	return nil
}

func (ms *memoryStorage) Delete(key string) error {
	if val, ok := ms.m[key]; ok {
		ms.usage -= int64(len(val))
		delete(ms.m, key)
	}
	return nil
}

func (ms *memoryStorage) EstimateQuota() (int64, int64, error) {
	return ms.quota, ms.usage, nil
}

type mockFetcher struct {
	data      []byte
	err       error
	fetchCount int
}

func (f *mockFetcher) Fetch(ctx context.Context, url string, onProgress func(done, total int64)) ([]byte, error) {
	f.fetchCount++
	if f.err != nil {
		return nil, f.err
	}
	if onProgress != nil {
		chunkSize := len(f.data) / 2
		if chunkSize == 0 {
			chunkSize = len(f.data)
		}
		onProgress(int64(chunkSize), int64(len(f.data)))
		if chunkSize < len(f.data) {
			onProgress(int64(len(f.data)), int64(len(f.data)))
		}
	}
	return f.data, nil
}

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
	// Mutate last byte of tensor data
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
			ptr := uintptr(reflect.ValueOf(&ts.Data[0]).Pointer())
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

	// Verify slice alias
	if &f32s[0] != (*float32)(reflect.ValueOf(&ts.Data[0]).UnsafePointer()) {
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

func TestLoad_CachesAfterFirstFetch(t *testing.T) {
	data := createTestArtifactData(t, "cache-model", 1)
	fetcher := &mockFetcher{data: data}
	storage := newMemoryStorage(1024 * 1024)

	cfg := weights.LoadConfig{
		ID:      "cache-model",
		Version: 1,
		URL:     "https://example.com/model.wtypw",
		Conn:    storage,
		Fetcher: fetcher,
	}

	// First load fetches from network
	art1, err := weights.Load(context.Background(), cfg)
	if err != nil {
		t.Fatalf("First Load failed: %v", err)
	}
	if art1.ID != "cache-model" {
		t.Errorf("got ID %s, want cache-model", art1.ID)
	}
	if fetcher.fetchCount != 1 {
		t.Errorf("fetcher count got %d, want 1", fetcher.fetchCount)
	}

	// Second load reads from cache
	art2, err := weights.Load(context.Background(), cfg)
	if err != nil {
		t.Fatalf("Second Load failed: %v", err)
	}
	if art2.ID != "cache-model" {
		t.Errorf("got ID %s, want cache-model", art2.ID)
	}
	if fetcher.fetchCount != 1 {
		t.Errorf("fetcher count got %d, want 1 (should be cached)", fetcher.fetchCount)
	}
}

func TestLoad_PartialBodyNotCached(t *testing.T) {
	data := createTestArtifactData(t, "partial-model", 1)
	truncated := data[:len(data)-10]
	fetcher := &mockFetcher{data: truncated}
	storage := newMemoryStorage(1024 * 1024)

	cfg := weights.LoadConfig{
		ID:      "partial-model",
		Version: 1,
		URL:     "https://example.com/partial.wtypw",
		Conn:    storage,
		Fetcher: fetcher,
	}

	_, err := weights.Load(context.Background(), cfg)
	if err == nil {
		t.Fatalf("expected error for partial body, got nil")
	}

	// Verify storage remains empty
	cached, _ := storage.Get("partial-model@1")
	if cached != nil {
		t.Errorf("expected cache to be empty, found %d bytes", len(cached))
	}
}

func TestLoad_VersionBumpRefetches(t *testing.T) {
	dataV1 := createTestArtifactData(t, "version-model", 1)
	dataV2 := createTestArtifactData(t, "version-model", 2)

	fetcher := &mockFetcher{data: dataV1}
	storage := newMemoryStorage(1024 * 1024)

	cfgV1 := weights.LoadConfig{
		ID:      "version-model",
		Version: 1,
		URL:     "https://example.com/version1.wtypw",
		Conn:    storage,
		Fetcher: fetcher,
	}

	_, err := weights.Load(context.Background(), cfgV1)
	if err != nil {
		t.Fatalf("Load V1 failed: %v", err)
	}
	if fetcher.fetchCount != 1 {
		t.Errorf("fetcher count got %d, want 1", fetcher.fetchCount)
	}

	// Version bump
	fetcher.data = dataV2
	cfgV2 := weights.LoadConfig{
		ID:      "version-model",
		Version: 2,
		URL:     "https://example.com/version2.wtypw",
		Conn:    storage,
		Fetcher: fetcher,
	}

	artV2, err := weights.Load(context.Background(), cfgV2)
	if err != nil {
		t.Fatalf("Load V2 failed: %v", err)
	}
	if artV2.Version != 2 {
		t.Errorf("got Version %d, want 2", artV2.Version)
	}
	if fetcher.fetchCount != 2 {
		t.Errorf("fetcher count got %d, want 2", fetcher.fetchCount)
	}
}

func TestLoad_ProgressCallback(t *testing.T) {
	data := createTestArtifactData(t, "progress-model", 1)
	fetcher := &mockFetcher{data: data}

	var progressReports [][2]int64
	onProgress := func(done, total int64) {
		progressReports = append(progressReports, [2]int64{done, total})
	}

	cfg := weights.LoadConfig{
		ID:         "progress-model",
		Version:    1,
		URL:        "https://example.com/progress.wtypw",
		Fetcher:    fetcher,
		OnProgress: onProgress,
	}

	_, err := weights.Load(context.Background(), cfg)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if len(progressReports) == 0 {
		t.Fatalf("expected progress reports, got none")
	}

	last := progressReports[len(progressReports)-1]
	if last[0] != last[1] || last[1] != int64(len(data)) {
		t.Errorf("last progress report got %v, want [%d %d]", last, len(data), len(data))
	}
}

func TestEvict(t *testing.T) {
	storage := newMemoryStorage(1024 * 1024)
	_ = storage.Put("model-to-evict@1", []byte("data"))

	if err := weights.Evict(storage, "model-to-evict@1"); err != nil {
		t.Fatalf("Evict failed: %v", err)
	}

	cached, _ := storage.Get("model-to-evict@1")
	if cached != nil {
		t.Errorf("expected entry to be evicted, got %v", cached)
	}
}

// Silence unused variable warning if any
var _ = fmt.Sprintf
var _ = crc32.ChecksumIEEE
var _ = binary.LittleEndian
