package weights_test

import (
	"bytes"
	"errors"
	"reflect"
	"testing"
	"unsafe"

	"webtyp.com/context"
	"webtyp.com/fetch"
	"webtyp.com/model"
	"webtyp.com/storage"
	"webtyp.com/weights"
)

type mockStorage struct {
	data      map[string][]byte
	quota     int64
	usage     int64
	lastQuery storage.Query
}

func newMockStorage(quota int64) *mockStorage {
	return &mockStorage{
		data:  make(map[string][]byte),
		quota: quota,
	}
}

func (m *mockStorage) EstimateQuota() (int64, int64, error) {
	if m.quota < 0 {
		return 0, 0, errors.New("quota unavailable")
	}
	return m.quota, m.usage, nil
}

func (m *mockStorage) Compile(q storage.Query, _ model.Model) (storage.Plan, error) {
	m.lastQuery = q
	return storage.Plan{}, nil
}

func (m *mockStorage) Exec(query string, args ...any) error {
	q := m.lastQuery
	switch q.Action {
	case storage.ActionCreate:
		key := q.Values[0].(string)
		val := q.Values[1].([]byte)
		m.data[key] = val
		m.usage += int64(len(val))
	case storage.ActionDelete:
		key := q.Conditions[0].Value().(string)
		if val, ok := m.data[key]; ok {
			m.usage -= int64(len(val))
			delete(m.data, key)
		}
	}
	return nil
}

func (m *mockStorage) QueryRow(query string, args ...any) storage.Scanner {
	q := m.lastQuery
	key := q.Conditions[0].Value().(string)
	val := m.data[key]
	return &mockScanner{val: val}
}

func (m *mockStorage) Query(query string, args ...any) (storage.Rows, error) {
	return nil, nil
}

func (m *mockStorage) Close() error {
	return nil
}

type mockScanner struct {
	val []byte
}

func (s *mockScanner) Scan(dest ...any) error {
	if len(s.val) == 0 {
		return storage.ErrNoRows
	}
	*(dest[0].(*[]byte)) = s.val
	return nil
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

func TestLoad_CachesAfterFirstFetch(t *testing.T) {
	data := createTestArtifactData(t, "cache-model", 1)
	fetchCount := 0

	mockFetcher := func(url string, cb func(*fetch.Response, error)) {
		fetchCount++
		resp := fetch.NewResponse(200, nil, data)
		cb(resp, nil)
	}

	storage := newMockStorage(1024 * 1024)
	cfg := weights.LoadConfig{
		ID:      "cache-model",
		Version: 1,
		URL:     "https://example.com/cache-model.wtypw",
		Conn:    storage,
		Fetcher: mockFetcher,
	}

	art1, err := weights.Load(context.Background(), cfg)
	if err != nil {
		t.Fatalf("First Load failed: %v", err)
	}
	if art1.ID != "cache-model" {
		t.Errorf("got ID %s, want cache-model", art1.ID)
	}
	if fetchCount != 1 {
		t.Errorf("fetchCount got %d, want 1", fetchCount)
	}

	art2, err := weights.Load(context.Background(), cfg)
	if err != nil {
		t.Fatalf("Second Load failed: %v", err)
	}
	if art2.ID != "cache-model" {
		t.Errorf("got ID %s, want cache-model", art2.ID)
	}
	if fetchCount != 1 {
		t.Errorf("fetchCount got %d, want 1 (should be cached)", fetchCount)
	}
}

func TestLoad_NoQuotaDoesNotCache(t *testing.T) {
	data := createTestArtifactData(t, "no-quota-model", 1)
	mockFetcher := func(url string, cb func(*fetch.Response, error)) {
		cb(fetch.NewResponse(200, nil, data), nil)
	}

	// Storage without quota estimation capability
	storage := newMockStorage(-1)
	cfg := weights.LoadConfig{
		ID:      "no-quota-model",
		Version: 1,
		URL:     "https://example.com/no-quota.wtypw",
		Conn:    storage,
		Fetcher: mockFetcher,
	}

	art, err := weights.Load(context.Background(), cfg)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if art.ID != "no-quota-model" {
		t.Errorf("got ID %s, want no-quota-model", art.ID)
	}

	if len(storage.data) != 0 {
		t.Errorf("expected storage to remain empty when quota estimation unavailable")
	}
}

func TestLoad_PartialBodyNotCached(t *testing.T) {
	data := createTestArtifactData(t, "partial-model", 1)
	truncated := data[:len(data)-10]

	mockFetcher := func(url string, cb func(*fetch.Response, error)) {
		cb(fetch.NewResponse(200, nil, truncated), nil)
	}

	storage := newMockStorage(1024 * 1024)
	cfg := weights.LoadConfig{
		ID:      "partial-model",
		Version: 1,
		URL:     "https://example.com/partial.wtypw",
		Conn:    storage,
		Fetcher: mockFetcher,
	}

	_, err := weights.Load(context.Background(), cfg)
	if err == nil {
		t.Fatalf("expected error for partial body, got nil")
	}

	if len(storage.data) != 0 {
		t.Errorf("expected storage to remain empty, found %d entries", len(storage.data))
	}
}

func TestLoad_VersionBumpRefetches(t *testing.T) {
	dataV1 := createTestArtifactData(t, "version-model", 1)
	dataV2 := createTestArtifactData(t, "version-model", 2)
	fetchCount := 0

	mockFetcher := func(url string, cb func(*fetch.Response, error)) {
		fetchCount++
		if fetchCount == 1 {
			cb(fetch.NewResponse(200, nil, dataV1), nil)
		} else {
			cb(fetch.NewResponse(200, nil, dataV2), nil)
		}
	}

	storage := newMockStorage(1024 * 1024)
	cfgV1 := weights.LoadConfig{
		ID:      "version-model",
		Version: 1,
		URL:     "https://example.com/version.wtypw",
		Conn:    storage,
		Fetcher: mockFetcher,
	}

	_, err := weights.Load(context.Background(), cfgV1)
	if err != nil {
		t.Fatalf("Load V1 failed: %v", err)
	}

	cfgV2 := weights.LoadConfig{
		ID:      "version-model",
		Version: 2,
		URL:     "https://example.com/version.wtypw",
		Conn:    storage,
		Fetcher: mockFetcher,
	}

	artV2, err := weights.Load(context.Background(), cfgV2)
	if err != nil {
		t.Fatalf("Load V2 failed: %v", err)
	}
	if artV2.Version != 2 {
		t.Errorf("got Version %d, want 2", artV2.Version)
	}
	if fetchCount != 2 {
		t.Errorf("fetchCount got %d, want 2", fetchCount)
	}
}

func TestEvict(t *testing.T) {
	storage := newMockStorage(1024 * 1024)
	key := weights.CacheKey("model-to-evict", 1)
	storage.data[key] = []byte("data")

	if err := weights.Evict(storage, "model-to-evict", 1); err != nil {
		t.Fatalf("Evict failed: %v", err)
	}

	if _, ok := storage.data[key]; ok {
		t.Errorf("expected entry to be evicted")
	}
}
