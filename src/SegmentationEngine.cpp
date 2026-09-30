#include "SegmentationEngine.h"
#include <opencv2/imgproc.hpp>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <array>
#include <stdexcept>

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

    const int seedBlock = (seed.y / blockSize) * bw + seed.x / blockSize;
    const int seedIntensity = graySource.at<uchar>(seed);
    if (blockMaxEdge[seedBlock] > edgePenaltyThreshold) {
        // A coarse boundary block needs pixel-level subdivision.
        regionGrowEdgeEmbedded(graySource, mask, seed, label, intensityThreshold, edgePenaltyThreshold, edges);
        return;
    }
    for (int i=0;i<numBlocks;++i)
        if (std::abs(blockMean[i] - seedIntensity) > intensityThreshold) blockMaxEdge[i] = 256;
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

    int seedRoot = ds.find(seedBlock);
    cv::Mat eligible = cv::Mat::zeros(graySource.size(), CV_8U);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        if(ds.find((y/blockSize)*bw+x/blockSize)==seedRoot &&
           edges.at<uchar>(y,x)<=edgePenaltyThreshold &&
           std::abs(int(graySource.at<uchar>(y,x))-seedIntensity)<=intensityThreshold)
            eligible.at<uchar>(y,x)=255;
    if(!eligible.at<uchar>(seed))return;
    cv::floodFill(eligible,seed,cv::Scalar(128),nullptr,cv::Scalar(),cv::Scalar(),4);
    mask.setTo(static_cast<uchar>(label),eligible==128);

}

void growCutFromSeeds(const cv::Mat& graySource, const cv::Mat& seeds,
                      cv::Mat& outMask, double beta, int maxIters, const std::function<bool()>& cancelled) {
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

    bool converged = false;
    for (int it = 0; it < maxIters; ++it) {
        if (cancelled && cancelled()) throw std::runtime_error("Segmentation cancelled");
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
        if (!changed) { converged = true; break; }
    }

    if (!converged) throw std::runtime_error("GrowCut did not converge. Add seeds or restrict the region.");
    if (std::any_of(strength.begin(), strength.end(), [](float value){return value<=0;}))
        throw std::runtime_error("GrowCut left unresolved pixels. Reduce edge sensitivity or add seeds.");
    outMask.create(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        uchar* m = outMask.ptr<uchar>(y);
        for (int x = 0; x < w; ++x)
            m[x] = label[static_cast<size_t>(y) * w + x];
    }
}

void randomWalkerFromSeeds(const cv::Mat& graySource, const cv::Mat& seeds,
                           cv::Mat& outMask, double beta, int maxIters, double tol,
                           const std::function<bool()>& cancelled) {
    CV_Assert(graySource.type() == CV_8UC1 && seeds.type() == CV_8UC1 && graySource.size() == seeds.size());
    const int w = graySource.cols, h = graySource.rows, n = w*h;
    std::vector<int> classes;
    for (int l=0;l<4;++l) if (cv::countNonZero(seeds == l)) classes.push_back(l);
    if (classes.size()<2) { outMask = cv::Mat(h,w,CV_8U,cv::Scalar(classes.empty()?0:classes[0])); return; }
    std::vector<int> freeIndex(n,-1), pixels;
    for(int i=0;i<n;++i) if(seeds.at<uchar>(i/w,i%w)==kNoSeed) {freeIndex[i]=pixels.size();pixels.push_back(i);}
    const int m=pixels.size();
    std::vector<std::array<int,4>> neighbors(m);
    std::vector<std::array<double,4>> weights(m);
    std::vector<double> diagonal(m,0), best(m,-1);
    cv::Mat result=seeds.clone();
    for(int j=0;j<m;++j) {
        int p=pixels[j],x=p%w,y=p/w;
        neighbors[j]={x>0?p-1:-1,x+1<w?p+1:-1,y>0?p-w:-1,y+1<h?p+w:-1};
        for(int k=0;k<4;++k) {
            int q=neighbors[j][k];if(q<0){weights[j][k]=0;continue;}
            double d=double(graySource.at<uchar>(y,x))-graySource.at<uchar>(q/w,q%w);
            weights[j][k]=std::max(1e-12,std::exp(-beta*d*d));diagonal[j]+=weights[j][k];
        }
    }
    auto multiply=[&](const std::vector<double>& x,std::vector<double>& y){
        for(int j=0;j<m;++j){y[j]=diagonal[j]*x[j];for(int k=0;k<4;++k){int q=neighbors[j][k];
            if(q>=0 && freeIndex[q]>=0)y[j]-=weights[j][k]*x[freeIndex[q]];}}
    };
    auto dot=[](const auto& a,const auto& b){double sum=0;for(size_t i=0;i<a.size();++i)sum+=a[i]*b[i];return sum;};
    for(int label:classes) {
        std::vector<double> b(m,0),x(m,0),r(m),z(m),p(m),ap(m);
        for(int j=0;j<m;++j) for(int k=0;k<4;++k){int q=neighbors[j][k];
            if(q>=0 && seeds.at<uchar>(q/w,q%w)==label)b[j]+=weights[j][k];}
        r=b;for(int j=0;j<m;++j)z[j]=r[j]/diagonal[j];p=z;
        double rz=dot(r,z),target=std::max(1e-24,dot(b,b)*tol*tol);
        bool converged=dot(r,r)<=target;
        for(int it=0;it<maxIters && !converged;++it){
            if(cancelled && cancelled()) throw std::runtime_error("Segmentation cancelled");
            multiply(p,ap);double denom=dot(p,ap);if(denom<=0 || !std::isfinite(denom))break;
            double alpha=rz/denom;
            for(int j=0;j<m;++j){x[j]+=alpha*p[j];r[j]-=alpha*ap[j];}
            if(dot(r,r)<=target){
                multiply(x,ap);for(int j=0;j<m;++j)r[j]=b[j]-ap[j];
                converged=dot(r,r)<=target;if(converged)break;
            }
            for(int j=0;j<m;++j)z[j]=r[j]/diagonal[j];
            double next=dot(r,z);if(!std::isfinite(next))break;
            for(int j=0;j<m;++j)p[j]=z[j]+(next/rz)*p[j];rz=next;
        }
        if(!converged)throw std::runtime_error("Random Walker did not converge. Use a smaller region or add seeds.");
        for(int j=0;j<m;++j)if(x[j]>best[j]){best[j]=x[j];int p=pixels[j];result.at<uchar>(p/w,p%w)=label;}
    }
    outMask=result;
}

bool graphCutFromSeeds(const cv::Mat& colorSource, const cv::Mat& seeds,
                       cv::Mat& mask, Label foreground, int iterations) {
    CV_Assert(colorSource.type() == CV_8UC3 && seeds.type() == CV_8UC1);
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(colorSource.size() == seeds.size() && colorSource.size() == mask.size());
    if (foreground == Label::Background) return false;
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
    try {
        cv::grabCut(colorSource, gc, cv::Rect(), bgModel, fgModel,
                    std::max(1, iterations), cv::GC_INIT_WITH_MASK);
    } catch (const cv::Exception&) {
        return false; // The caller's mask has not been modified yet.
    }

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
