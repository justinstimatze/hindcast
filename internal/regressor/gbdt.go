package regressor

import (
	"math"
	"sort"
)

// Tree is one regression tree. Nodes are stored as a flat slice; root is
// index 0. Leaves carry a value, internal nodes carry a feature/threshold
// and indices of left/right children.
type Tree struct {
	Nodes []Node
}

// Node is either a leaf (Leaf=true, LeafValue set) or a split (Leaf=false,
// FeatureIdx/Threshold/Left/Right set). "feats[FeatureIdx] <= Threshold"
// goes Left, else Right. Single-pass evaluation, no allocation.
type Node struct {
	Leaf       bool
	LeafValue  float64
	FeatureIdx int
	Threshold  float64
	Left       int
	Right      int
}

// Predict walks the tree for one feature row. Returns 0 if the tree is
// empty or if a node's FeatureIdx exceeds the runtime feature vector
// length (defensive against a stale on-disk model from a prior feature
// count). Without the bounds check, a single dimension-mismatched tree
// panics out the entire regressor tier in production.
func (t *Tree) Predict(feats []float64) float64 {
	if len(t.Nodes) == 0 {
		return 0
	}
	idx := 0
	for {
		n := t.Nodes[idx]
		if n.Leaf {
			return n.LeafValue
		}
		if n.FeatureIdx < 0 || n.FeatureIdx >= len(feats) {
			return 0
		}
		if feats[n.FeatureIdx] <= n.Threshold {
			idx = n.Left
		} else {
			idx = n.Right
		}
	}
}

// Params controls boosting. Defaults: 100 rounds, depth-3 trees, lr=0.1,
// minSamplesLeaf=5. Targets typical hindcast corpus size (hundreds to
// low-thousands of records); larger / deeper would overfit.
type Params struct {
	NumRounds      int
	MaxDepth       int
	LR             float64
	MinSamplesLeaf int
}

func DefaultParams() Params {
	return Params{
		NumRounds:      100,
		MaxDepth:       3,
		LR:             0.1,
		MinSamplesLeaf: 5,
	}
}

// trainGBDT fits an additive tree ensemble on (X, y) under squared loss.
// Initial prediction is mean(y). Each round fits a tree to the residuals
// (y - current_prediction) and adds LR × tree to the running prediction.
func trainGBDT(X [][]float64, y []float64, p Params) (*Model, error) {
	if len(X) == 0 || len(X) != len(y) {
		return nil, errEmpty
	}
	nFeat := len(X[0])

	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(len(y))

	m := &Model{
		BaseValue:    mean,
		LR:           p.LR,
		Trees:        make([]Tree, 0, p.NumRounds),
		FeatureNames: append([]string(nil), FeatureNames...),
	}
	preds := make([]float64, len(y))
	for i := range preds {
		preds[i] = mean
	}
	residuals := make([]float64, len(y))

	for round := 0; round < p.NumRounds; round++ {
		for i := range y {
			residuals[i] = y[i] - preds[i]
		}
		tree := buildTree(X, residuals, p.MaxDepth, nFeat, p.MinSamplesLeaf)
		// Skip empty / no-split trees — they contribute nothing.
		if len(tree.Nodes) == 1 && tree.Nodes[0].Leaf && tree.Nodes[0].LeafValue == 0 {
			continue
		}
		m.Trees = append(m.Trees, tree)
		for i := range preds {
			preds[i] += p.LR * tree.Predict(X[i])
		}
	}
	return m, nil
}

// buildTree fits one regression tree by recursive best-split search.
func buildTree(X [][]float64, y []float64, maxDepth, nFeat, minLeaf int) Tree {
	indices := make([]int, len(y))
	for i := range indices {
		indices[i] = i
	}
	var nodes []Node
	buildNode(X, y, indices, 0, maxDepth, nFeat, minLeaf, &nodes)
	return Tree{Nodes: nodes}
}

// buildNode appends to *nodes and returns the new node's index. Reserves
// the parent slot first (so child recursion can safely append) and fills
// it in after recursion.
func buildNode(X [][]float64, y []float64, indices []int, depth, maxDepth, nFeat, minLeaf int, nodes *[]Node) int {
	myIdx := len(*nodes)
	*nodes = append(*nodes, Node{}) // placeholder

	parentSum := 0.0
	parentSumSq := 0.0
	for _, i := range indices {
		parentSum += y[i]
		parentSumSq += y[i] * y[i]
	}
	parentMean := parentSum / float64(len(indices))

	if depth >= maxDepth || len(indices) < 2*minLeaf {
		(*nodes)[myIdx] = Node{Leaf: true, LeafValue: parentMean}
		return myIdx
	}

	parentVar := parentSumSq/float64(len(indices)) - parentMean*parentMean
	if parentVar <= 0 {
		(*nodes)[myIdx] = Node{Leaf: true, LeafValue: parentMean}
		return myIdx
	}

	bestFeat := -1
	bestThresh := 0.0
	bestGain := 0.0
	var bestLeft, bestRight []int

	type vi struct {
		v float64
		i int
	}
	scratch := make([]vi, len(indices))

	for f := 0; f < nFeat; f++ {
		for k, i := range indices {
			scratch[k] = vi{X[i][f], i}
		}
		sort.Slice(scratch, func(a, b int) bool { return scratch[a].v < scratch[b].v })

		leftSum, leftSumSq := 0.0, 0.0
		rightSum, rightSumSq := parentSum, parentSumSq
		for k := 0; k < len(scratch)-1; k++ {
			yi := y[scratch[k].i]
			leftSum += yi
			leftSumSq += yi * yi
			rightSum -= yi
			rightSumSq -= yi * yi

			if scratch[k].v == scratch[k+1].v {
				continue
			}
			nL := k + 1
			nR := len(scratch) - nL
			if nL < minLeaf || nR < minLeaf {
				continue
			}
			lMean := leftSum / float64(nL)
			rMean := rightSum / float64(nR)
			lVar := leftSumSq/float64(nL) - lMean*lMean
			rVar := rightSumSq/float64(nR) - rMean*rMean
			if lVar < 0 {
				lVar = 0
			}
			if rVar < 0 {
				rVar = 0
			}
			gain := parentVar - (float64(nL)*lVar+float64(nR)*rVar)/float64(len(indices))
			if gain > bestGain {
				bestGain = gain
				bestFeat = f
				bestThresh = (scratch[k].v + scratch[k+1].v) / 2
				bestLeft = make([]int, 0, nL)
				bestRight = make([]int, 0, nR)
				for j := 0; j <= k; j++ {
					bestLeft = append(bestLeft, scratch[j].i)
				}
				for j := k + 1; j < len(scratch); j++ {
					bestRight = append(bestRight, scratch[j].i)
				}
			}
		}
	}

	if bestFeat < 0 {
		(*nodes)[myIdx] = Node{Leaf: true, LeafValue: parentMean}
		return myIdx
	}

	leftIdx := buildNode(X, y, bestLeft, depth+1, maxDepth, nFeat, minLeaf, nodes)
	rightIdx := buildNode(X, y, bestRight, depth+1, maxDepth, nFeat, minLeaf, nodes)
	(*nodes)[myIdx] = Node{
		FeatureIdx: bestFeat,
		Threshold:  bestThresh,
		Left:       leftIdx,
		Right:      rightIdx,
	}
	return myIdx
}

// MAD computes mean absolute log-ratio of (predicted, actual) pairs in
// the original (un-logged) space — diagnostic helper used by trainers
// to print a quality estimate.
func MAD(predictedLog, actualLog []float64) float64 {
	if len(predictedLog) == 0 {
		return 0
	}
	sum := 0.0
	for i := range predictedLog {
		sum += math.Abs(predictedLog[i] - actualLog[i])
	}
	return math.Exp(sum / float64(len(predictedLog)))
}
