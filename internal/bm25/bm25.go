// Package bm25 provides a pure-stdlib BM25 inverted index with
// incremental update and top-k scoring. Gob-encoded on-disk format.
//
// Privacy by construction: tokens are hashed with a per-install salt
// before entering the index, so the on-disk index contains no prompt
// text. The salt is generated at install time and stored at
// ~/.claude/hindcast/salt (0600). Queries are hashed with the same salt
// and looked up by hash. The index never sees plaintext tokens except
// ephemerally in memory during Add/TopK.
//
// Scoring follows the standard BM25 formulation with k1=1.2, b=0.75.
// Similarity is normalized against the query's self-score so a value
// of 1.0 represents a hypothetical perfect match.
package bm25

import (
	"encoding/binary"
	"encoding/gob"
	"errors"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

type Posting struct {
	DocID uint32
	TF    uint16
}

// Doc is the display metadata associated with an indexed prompt. Notably
// does NOT contain the prompt text — only deterministic-derived fields
// (task_type, size_bucket, durations, tool/file counts). This is the
// primary privacy invariant: prompts never persist.
type Doc struct {
	ID            uint32
	Length        int // token count
	ActiveSeconds int
	WallSeconds   int
	TaskType      string
	SizeBucket    string
	ToolCount     int
	FilesTouched  int
}

type Index struct {
	// Postings: salted-token-hash → posting list sorted by DocID ascending.
	Postings map[uint64][]Posting
	Docs     []Doc
	TotalLen int
	AvgDocLen float64
}

type Match struct {
	Doc Doc
	Sim float64
}

func New() *Index {
	return &Index{Postings: map[uint64][]Posting{}}
}

// Load reads a gob-encoded index. Returns an empty index if the file
// doesn't exist.
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return New(), nil
		}
		return nil, err
	}
	defer f.Close()
	idx := &Index{}
	if err := gob.NewDecoder(f).Decode(idx); err != nil {
		return nil, err
	}
	if idx.Postings == nil {
		idx.Postings = map[uint64][]Posting{}
	}
	return idx, nil
}

func (i *Index) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".bm25-*.gob")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := func() {
		tmp.Close()
		os.Remove(tmpName)
	}
	if err := tmp.Chmod(0600); err != nil {
		cleanup()
		return err
	}
	if err := gob.NewEncoder(tmp).Encode(i); err != nil {
		cleanup()
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, path)
}

// Add indexes a new document via its salted token hashes. Caller
// tokenizes+hashes; the index never sees plaintext.
func (i *Index) Add(hashes []uint64, meta Doc) uint32 {
	if i.Postings == nil {
		i.Postings = map[uint64][]Posting{}
	}
	meta.ID = uint32(len(i.Docs))
	meta.Length = len(hashes)

	tf := map[uint64]int{}
	for _, h := range hashes {
		tf[h]++
	}
	for h, count := range tf {
		if count > math.MaxUint16 {
			count = math.MaxUint16
		}
		i.Postings[h] = append(i.Postings[h], Posting{
			DocID: meta.ID,
			TF:    uint16(count),
		})
	}

	i.Docs = append(i.Docs, meta)
	i.TotalLen += meta.Length
	if len(i.Docs) > 0 {
		i.AvgDocLen = float64(i.TotalLen) / float64(len(i.Docs))
	}
	return meta.ID
}

// TopK returns up to k docs scored against query hashes, highest sim first.
func (i *Index) TopK(queryHashes []uint64, k int) []Match {
	if k <= 0 || len(i.Docs) == 0 || len(queryHashes) == 0 {
		return nil
	}
	qTF := map[uint64]int{}
	for _, h := range queryHashes {
		qTF[h]++
	}

	N := float64(len(i.Docs))
	avgdl := i.AvgDocLen
	if avgdl == 0 {
		avgdl = 1
	}

	scores := map[uint32]float64{}
	for h := range qTF {
		postings := i.Postings[h]
		if len(postings) == 0 {
			continue
		}
		idf := math.Log((N-float64(len(postings))+0.5)/(float64(len(postings))+0.5) + 1)
		for _, p := range postings {
			doc := i.Docs[p.DocID]
			dl := float64(doc.Length)
			if dl == 0 {
				dl = 1
			}
			numer := float64(p.TF) * (bm25K1 + 1)
			denom := float64(p.TF) + bm25K1*(1-bm25B+bm25B*(dl/avgdl))
			scores[p.DocID] += idf * numer / denom
		}
	}
	if len(scores) == 0 {
		return nil
	}

	type scored struct {
		id uint32
		s  float64
	}
	list := make([]scored, 0, len(scores))
	for id, s := range scores {
		list = append(list, scored{id, s})
	}
	sort.Slice(list, func(a, b int) bool { return list[a].s > list[b].s })

	if len(list) > k {
		list = list[:k]
	}

	self := i.selfScore(qTF, len(queryHashes))
	matches := make([]Match, 0, len(list))
	for _, s := range list {
		sim := 0.0
		if self > 0 {
			sim = s.s / self
			if sim > 1 {
				sim = 1
			}
		}
		matches = append(matches, Match{Doc: i.Docs[s.id], Sim: sim})
	}
	return matches
}

func (i *Index) selfScore(qTF map[uint64]int, qLen int) float64 {
	if len(i.Docs) == 0 {
		return 0
	}
	N := float64(len(i.Docs))
	avgdl := i.AvgDocLen
	if avgdl == 0 {
		avgdl = 1
	}
	dl := float64(qLen)
	if dl == 0 {
		dl = 1
	}
	total := 0.0
	for h, tf := range qTF {
		df := float64(len(i.Postings[h]))
		if df == 0 {
			continue
		}
		idf := math.Log((N-df+0.5)/(df+0.5) + 1)
		numer := float64(tf) * (bm25K1 + 1)
		denom := float64(tf) + bm25K1*(1-bm25B+bm25B*(dl/avgdl))
		total += idf * numer / denom
	}
	return total
}

// Hash returns a 64-bit salted FNV-1a hash for a single token. The salt
// comes from ~/.claude/hindcast/salt, initialized at install time.
func Hash(token string, salt []byte) uint64 {
	h := fnv.New64a()
	h.Write(salt)
	var lenPrefix [4]byte
	binary.LittleEndian.PutUint32(lenPrefix[:], uint32(len(token)))
	h.Write(lenPrefix[:])
	h.Write([]byte(token))
	return h.Sum64()
}

// HashTokens tokenizes a prompt (lowercases, strips non-alphanumeric,
// drops stopwords and single-character tokens) and returns the salted
// hash of each remaining token. Stable across calls.
func HashTokens(prompt string, salt []byte) []uint64 {
	toks := tokenize(prompt)
	out := make([]uint64, 0, len(toks))
	for _, t := range toks {
		out = append(out, Hash(t, salt))
	}
	return out
}

func tokenize(s string) []string {
	var toks []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() == 0 {
			return
		}
		t := cur.String()
		cur.Reset()
		if len(t) < 2 {
			return
		}
		if stopwords[t] {
			return
		}
		toks = append(toks, t)
	}
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return toks
}

var stopwords = map[string]bool{
	"the": true, "an": true,
	"is": true, "are": true, "was": true, "were": true,
	"be": true, "been": true, "being": true,
	"of": true, "to": true, "in": true, "on": true, "at": true,
	"for": true, "with": true, "from": true, "by": true,
	"and": true, "or": true, "but": true, "if": true, "then": true,
	"that": true, "this": true, "these": true, "those": true,
	"it": true, "its": true,
	"me": true, "my": true, "we": true, "us": true, "our": true,
	"do": true, "does": true, "did": true, "done": true,
}
