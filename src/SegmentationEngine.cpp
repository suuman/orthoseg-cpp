#include "SegmentationEngine.h"
#include <opencv2/imgproc.hpp>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>

namespace orthoseg {

cv::Mat computeEdgeMap(const cv::Mat& graySource) {
    CV_Assert(graySource.type() == CV_8UC1);
    cv::Mat gx, gy;
    cv::Sobel(graySource, gx, CV_32F, 1, 0, 3);
    cv::Sobel(graySource, gy, CV_32F, 0, 1, 3);
    cv::Mat mag;
    cv::magnitude(gx, gy, mag);          // sqrt(gx^2 + gy^2)
    cv::Mat edges;
    mag.convertTo(edges, CV_8U);         // saturating cast clamps to 255
    // The reference leaves a 1-pixel zero border (its Sobel loop skips the
    // image boundary), so border pixels never block region growth.
    if (edges.rows > 1 && edges.cols > 1) {
        edges.row(0).setTo(0);
        edges.row(edges.rows - 1).setTo(0);
        edges.col(0).setTo(0);
        edges.col(edges.cols - 1).setTo(0);
    }
    return edges;
}

static inline bool inBounds(const cv::Mat& m, int x, int y) {
    return x >= 0 && y >= 0 && x < m.cols && y < m.rows;
}

void regionGrowStandard(const cv::Mat& graySource, cv::Mat& mask,
                        cv::Point seed, Label label, int intensityThreshold) {
    CV_Assert(graySource.type() == CV_8UC1 && mask.type() == CV_8UC1);
    CV_Assert(graySource.size() == mask.size());
    if (!inBounds(graySource, seed.x, seed.y)) return;

    const int w = graySource.cols, h = graySource.rows;
    const uchar labelId = static_cast<uchar>(label);
    const int seedIntensity = graySource.at<uchar>(seed);

    std::vector<char> visited(static_cast<size_t>(w) * h, 0);
    std::queue<cv::Point> q;
    q.push(seed);
    visited[seed.y * w + seed.x] = 1;

    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = {0, 0, -1, 1};

    while (!q.empty()) {
        cv::Point p = q.front();
        q.pop();
        if (std::abs(graySource.at<uchar>(p) - seedIntensity) > intensityThreshold)
            continue;
        mask.at<uchar>(p) = labelId;
        for (int k = 0; k < 4; ++k) {
            int nx = p.x + dx[k], ny = p.y + dy[k];
            if (!inBounds(graySource, nx, ny)) continue;
            size_t ni = static_cast<size_t>(ny) * w + nx;
            if (visited[ni]) continue;
            visited[ni] = 1;
            q.push({nx, ny});
        }
    }
}

void regionGrowEdgeEmbedded(const cv::Mat& graySource, cv::Mat& mask,
                            cv::Point seed, Label label,
                            int intensityThreshold, int edgePenaltyThreshold,
                            const cv::Mat& edgeMapIn) {
    CV_Assert(graySource.type() == CV_8UC1 && mask.type() == CV_8UC1);
    CV_Assert(graySource.size() == mask.size());
    if (!inBounds(graySource, seed.x, seed.y)) return;

    cv::Mat edges = edgeMapIn.empty() ? computeEdgeMap(graySource) : edgeMapIn;
    const int w = graySource.cols, h = graySource.rows;
    const uchar labelId = static_cast<uchar>(label);
    const int seedIntensity = graySource.at<uchar>(seed);

    std::vector<char> visited(static_cast<size_t>(w) * h, 0);
    std::queue<cv::Point> q;
    q.push(seed);
    visited[seed.y * w + seed.x] = 1;

    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = {0, 0, -1, 1};

    while (!q.empty()) {
        cv::Point p = q.front();
        q.pop();
        // Halt growth at strong boundaries.
        if (edges.at<uchar>(p) > edgePenaltyThreshold) continue;
        if (std::abs(graySource.at<uchar>(p) - seedIntensity) > intensityThreshold)
            continue;
        mask.at<uchar>(p) = labelId;
        for (int k = 0; k < 4; ++k) {
            int nx = p.x + dx[k], ny = p.y + dy[k];
            if (!inBounds(graySource, nx, ny)) continue;
            size_t ni = static_cast<size_t>(ny) * w + nx;
            if (visited[ni]) continue;
            visited[ni] = 1;
            q.push({nx, ny});
        }
    }
}

namespace {
struct DisjointSet {
    std::vector<int> parent;
    explicit DisjointSet(int n) : parent(n) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]]; // path halving
            i = parent[i];
        }
        return i;
    }
    void unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra != rb) parent[ra] = rb;
    }
};
} // namespace

void regionGrowSplitMerge(const cv::Mat& graySource, cv::Mat& mask,
                          cv::Point seed, Label label,
                          int intensityThreshold, int edgePenaltyThreshold,
                          int blockSize, const cv::Mat& edgeMapIn) {
    CV_Assert(graySource.type() == CV_8UC1 && mask.type() == CV_8UC1);
    CV_Assert(graySource.size() == mask.size());
    if (!inBounds(graySource, seed.x, seed.y)) return;
    if (blockSize < 1) blockSize = 1;

    cv::Mat edges = edgeMapIn.empty() ? computeEdgeMap(graySource) : edgeMapIn;
    const int w = graySource.cols, h = graySource.rows;
    const int bw = (w + blockSize - 1) / blockSize;
    const int bh = (h + blockSize - 1) / blockSize;
    const int numBlocks = bw * bh;

    std::vector<float> blockMean(numBlocks, 0.f);
    std::vector<int>   blockMaxEdge(numBlocks, 0);

    // Feature extraction: mean intensity and max edge magnitude per block.
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            long sum = 0; int count = 0, maxEdge = 0;
            int y0 = by * blockSize, y1 = std::min((by + 1) * blockSize, h);
            int x0 = bx * blockSize, x1 = std::min((bx + 1) * blockSize, w);
            for (int y = y0; y < y1; ++y) {
                const uchar* srow = graySource.ptr<uchar>(y);
                const uchar* erow = edges.ptr<uchar>(y);
                for (int x = x0; x < x1; ++x) {
                    sum += srow[x];
                    if (erow[x] > maxEdge) maxEdge = erow[x];
                    ++count;
                }
            }
            int bi = by * bw + bx;
            blockMean[bi] = count ? static_cast<float>(sum) / count : 0.f;
            blockMaxEdge[bi] = maxEdge;
        }
    }

    // Merge adjacent blocks: both below the edge penalty and similar in mean.
    DisjointSet ds(numBlocks);
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            int u = by * bw + bx;
            if (bx < bw - 1) {
                int v = by * bw + (bx + 1);
                if (blockMaxEdge[u] <= edgePenaltyThreshold &&
                    blockMaxEdge[v] <= edgePenaltyThreshold &&
                    std::abs(blockMean[u] - blockMean[v]) <= intensityThreshold)
                    ds.unite(u, v);
            }
            if (by < bh - 1) {
                int v = (by + 1) * bw + bx;
                if (blockMaxEdge[u] <= edgePenaltyThreshold &&
                    blockMaxEdge[v] <= edgePenaltyThreshold &&
                    std::abs(blockMean[u] - blockMean[v]) <= intensityThreshold)
                    ds.unite(u, v);
            }
        }
    }

    // Fill every block sharing the seed block's root.
    int seedRoot = ds.find((seed.y / blockSize) * bw + (seed.x / blockSize));
    const uchar labelId = static_cast<uchar>(label);
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            if (ds.find(by * bw + bx) != seedRoot) continue;
            int y0 = by * blockSize, y1 = std::min((by + 1) * blockSize, h);
            int x0 = bx * blockSize, x1 = std::min((bx + 1) * blockSize, w);
            for (int y = y0; y < y1; ++y) {
                uchar* mrow = mask.ptr<uchar>(y);
                for (int x = x0; x < x1; ++x) mrow[x] = labelId;
            }
        }
    }
}

void growCutFromSeeds(const cv::Mat& graySource, const cv::Mat& seeds,
                      cv::Mat& outMask, double beta, int maxIters) {
    CV_Assert(graySource.type() == CV_8UC1 && seeds.type() == CV_8UC1);
    CV_Assert(graySource.size() == seeds.size());
    const int w = graySource.cols, h = graySource.rows;
    const size_t N = static_cast<size_t>(w) * h;

    std::vector<int>   I(N);
    std::vector<uchar> label(N), nextLabel(N);
    std::vector<float> strength(N, 0.f), nextStrength(N, 0.f);

    for (int y = 0; y < h; ++y) {
        const uchar* g = graySource.ptr<uchar>(y);
        const uchar* s = seeds.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            size_t i = static_cast<size_t>(y) * w + x;
            I[i] = g[x];
            if (s[x] != kNoSeed) { label[i] = s[x]; strength[i] = 1.f; }
            else                 { label[i] = 0;    strength[i] = 0.f; }
        }
    }

    auto g = [beta](int a, int b) {
        double d = static_cast<double>(a) - b;
        return std::exp(-beta * d * d);
    };

    if (maxIters <= 0) maxIters = w + h; // enough for a label to cross the image
    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = {0, 0, -1, 1};

    for (int it = 0; it < maxIters; ++it) {
        // Synchronous update: every cell reads the previous state.
        nextLabel = label;
        nextStrength = strength;
        bool changed = false;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t i = static_cast<size_t>(y) * w + x;
                float best = strength[i];
                uchar bestLabel = label[i];
                for (int k = 0; k < 4; ++k) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    size_t j = static_cast<size_t>(ny) * w + nx;
                    if (strength[j] <= 0.f) continue;
                    float attack = static_cast<float>(g(I[i], I[j])) * strength[j];
                    if (attack > best) { best = attack; bestLabel = label[j]; }
                }
                if (best > strength[i]) {   // strengths only increase → converges
                    nextStrength[i] = best;
                    nextLabel[i] = bestLabel;
                    changed = true;
                }
            }
        }
        std::swap(label, nextLabel);
        std::swap(strength, nextStrength);
        if (!changed) break;
    }

    outMask.create(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        uchar* m = outMask.ptr<uchar>(y);
        for (int x = 0; x < w; ++x)
            m[x] = label[static_cast<size_t>(y) * w + x];
    }
}

void randomWalkerFromSeeds(const cv::Mat& graySource, const cv::Mat& seeds,
                           cv::Mat& outMask, double beta,
                           int maxIters, double tol) {
    CV_Assert(graySource.type() == CV_8UC1 && seeds.type() == CV_8UC1);
    CV_Assert(graySource.size() == seeds.size());
    const int w = graySource.cols, h = graySource.rows;
    const size_t N = static_cast<size_t>(w) * h;

    // Which labels are present among the seeds?
    int fieldOf[4] = {-1, -1, -1, -1};
    std::vector<uchar> labelList;
    for (int y = 0; y < h; ++y) {
        const uchar* s = seeds.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            uchar v = s[x];
            if (v != kNoSeed && v < 4 && fieldOf[v] < 0) {
                fieldOf[v] = 1; // mark; index assigned below
            }
        }
    }
    for (int l = 0; l < 4; ++l)
        if (fieldOf[l] >= 0) { fieldOf[l] = static_cast<int>(labelList.size());
                               labelList.push_back(static_cast<uchar>(l)); }
    const int K = static_cast<int>(labelList.size());

    outMask.create(h, w, CV_8UC1);
    if (K == 0) { outMask.setTo(0); return; }
    if (K == 1) { outMask.setTo(labelList[0]); return; }

    // Per-pixel: seed field index, or -1 if free.
    std::vector<int>   seedField(N, -1);
    std::vector<float> I(N);
    for (int y = 0; y < h; ++y) {
        const uchar* g = graySource.ptr<uchar>(y);
        const uchar* s = seeds.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            size_t i = static_cast<size_t>(y) * w + x;
            I[i] = g[x];
            if (s[x] != kNoSeed && s[x] < 4) seedField[i] = fieldOf[s[x]];
        }
    }

    // Edge weights to the right/down neighbour: w = exp(-beta*(dI)^2).
    std::vector<double> wR(N, 0.0), wD(N, 0.0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t i = static_cast<size_t>(y) * w + x;
            if (x < w - 1) { double d = I[i] - I[i + 1];     wR[i] = std::exp(-beta * d * d); }
            if (y < h - 1) { double d = I[i] - I[i + w];     wD[i] = std::exp(-beta * d * d); }
        }
    }

    // Potentials U[i*K + k]: seeds pinned, free pixels start uniform.
    std::vector<double> U(N * K, 0.0);
    for (size_t i = 0; i < N; ++i) {
        if (seedField[i] >= 0) U[i * K + seedField[i]] = 1.0;
        else for (int k = 0; k < K; ++k) U[i * K + k] = 1.0 / K;
    }

    // SOR (Gauss-Seidel with over-relaxation) on the harmonic system.
    const double omega = 1.8;
    for (int it = 0; it < maxIters; ++it) {
        double maxDelta = 0.0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t i = static_cast<size_t>(y) * w + x;
                if (seedField[i] >= 0) continue;
                double wl = (x > 0)     ? wR[i - 1] : 0.0;
                double wr = (x < w - 1) ? wR[i]     : 0.0;
                double wu = (y > 0)     ? wD[i - w] : 0.0;
                double wd = (y < h - 1) ? wD[i]     : 0.0;
                double wsum = wl + wr + wu + wd;
                if (wsum <= 0.0) continue;
                for (int k = 0; k < K; ++k) {
                    double num = 0.0;
                    if (wl) num += wl * U[(i - 1) * K + k];
                    if (wr) num += wr * U[(i + 1) * K + k];
                    if (wu) num += wu * U[(i - w) * K + k];
                    if (wd) num += wd * U[(i + w) * K + k];
                    double newv = num / wsum;
                    double old = U[i * K + k];
                    double upd = old + omega * (newv - old);
                    U[i * K + k] = upd;
                    double delta = std::abs(upd - old);
                    if (delta > maxDelta) maxDelta = delta;
                }
            }
        }
        if (maxDelta < tol) break;
    }

    for (int y = 0; y < h; ++y) {
        uchar* m = outMask.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            size_t i = static_cast<size_t>(y) * w + x;
            int bestK = 0; double bestV = U[i * K];
            for (int k = 1; k < K; ++k)
                if (U[i * K + k] > bestV) { bestV = U[i * K + k]; bestK = k; }
            m[x] = labelList[bestK];
        }
    }
}

bool graphCutFromSeeds(const cv::Mat& colorSource, const cv::Mat& seeds,
                       cv::Mat& mask, Label foreground, int iterations) {
    CV_Assert(colorSource.type() == CV_8UC3 && seeds.type() == CV_8UC1);
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(colorSource.size() == seeds.size() && colorSource.size() == mask.size());
    const uchar fg = static_cast<uchar>(foreground);

    cv::Mat gc(colorSource.size(), CV_8UC1);
    int nFG = 0, nBG = 0;
    for (int y = 0; y < gc.rows; ++y) {
        const uchar* s = seeds.ptr<uchar>(y);
        uchar* g = gc.ptr<uchar>(y);
        for (int x = 0; x < gc.cols; ++x) {
            if (s[x] == kNoSeed)      g[x] = cv::GC_PR_BGD;      // unseeded: probably bg
            else if (s[x] == fg)    { g[x] = cv::GC_FGD; ++nFG; } // hard foreground
            else                    { g[x] = cv::GC_BGD; ++nBG; } // hard background
        }
    }
    if (nFG == 0 || nBG == 0) return false; // grabCut needs both classes

    cv::Mat bgModel, fgModel;
    cv::grabCut(colorSource, gc, cv::Rect(), bgModel, fgModel,
                std::max(1, iterations), cv::GC_INIT_WITH_MASK);

    for (int y = 0; y < gc.rows; ++y) {
        const uchar* g = gc.ptr<uchar>(y);
        uchar* m = mask.ptr<uchar>(y);
        for (int x = 0; x < gc.cols; ++x) {
            bool isFg = (g[x] & 1); // GC_FGD(1)/GC_PR_FGD(3) are odd
            if (isFg)              m[x] = fg;
            else if (m[x] == fg)   m[x] = 0; // cut says bg: retract this label
        }
    }
    return true;
}

} // namespace orthoseg
