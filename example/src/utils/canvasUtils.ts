/**
 * Flood fill algorithm for canvas
 */
export function floodFill(
  ctx: CanvasRenderingContext2D,
  startX: number,
  startY: number,
  fillColor: { r: number; g: number; b: number; a: number }
) {
  const canvas = ctx.canvas;
  const width = canvas.width;
  const height = canvas.height;
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;

  const startPos = (startY * width + startX) * 4;
  const startR = data[startPos];
  const startG = data[startPos + 1];
  const startB = data[startPos + 2];
  const startA = data[startPos + 3];

  // If already the same color, skip
  if (
    startR === fillColor.r &&
    startG === fillColor.g &&
    startB === fillColor.b &&
    startA === fillColor.a
  ) {
    return;
  }

  const stack: [number, number][] = [[startX, startY]];

  while (stack.length > 0) {
    const [x, y] = stack.pop()!;
    const pos = (y * width + x) * 4;

    if (
      data[pos] === startR &&
      data[pos + 1] === startG &&
      data[pos + 2] === startB &&
      data[pos + 3] === startA
    ) {
      data[pos] = fillColor.r;
      data[pos + 1] = fillColor.g;
      data[pos + 2] = fillColor.b;
      data[pos + 3] = fillColor.a;

      if (x > 0) stack.push([x - 1, y]);
      if (x < width - 1) stack.push([x + 1, y]);
      if (y > 0) stack.push([x, y - 1]);
      if (y < height - 1) stack.push([x, y + 1]);
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

export function computeEdgeMap(mainData: Uint8ClampedArray, width: number, height: number): Uint8Array {
  const edges = new Uint8Array(width * height);
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    gray[i] = (mainData[i * 4] + mainData[i * 4 + 1] + mainData[i * 4 + 2]) / 3;
  }
  
  const Gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const Gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sumX = 0;
      let sumY = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const val = gray[(y + ky) * width + (x + kx)];
          const idx = (ky + 1) * 3 + (kx + 1);
          sumX += val * Gx[idx];
          sumY += val * Gy[idx];
        }
      }
      const mag = Math.sqrt(sumX * sumX + sumY * sumY);
      edges[y * width + x] = Math.min(255, mag);
    }
  }
  return edges;
}

/**
 * Standard Region growing fill algorithm based on underlaying image intensity.
 */
export function regionGrowFillStandard(
  mainCtx: CanvasRenderingContext2D,
  maskCtx: CanvasRenderingContext2D,
  startX: number,
  startY: number,
  fillColor: { r: number; g: number; b: number; a: number },
  threshold: number
) {
  const width = mainCtx.canvas.width;
  const height = mainCtx.canvas.height;
  
  const mainImageData = mainCtx.getImageData(0, 0, width, height);
  const mainData = mainImageData.data;
  
  const maskImageData = maskCtx.getImageData(0, 0, width, height);
  const maskData = maskImageData.data;

  const visited = new Uint8Array(width * height);

  const startPos = (startY * width + startX) * 4;
  const seedIntensity = (mainData[startPos] + mainData[startPos + 1] + mainData[startPos + 2]) / 3;

  const stack: [number, number][] = [[startX, startY]];

  while (stack.length > 0) {
    const [x, y] = stack.pop()!;
    const idx = y * width + x;
    
    if (visited[idx]) continue;
    visited[idx] = 1;

    const pos = idx * 4;
    
    const intensity = (mainData[pos] + mainData[pos + 1] + mainData[pos + 2]) / 3;

    if (Math.abs(intensity - seedIntensity) <= threshold) {
      maskData[pos] = fillColor.r;
      maskData[pos + 1] = fillColor.g;
      maskData[pos + 2] = fillColor.b;
      maskData[pos + 3] = fillColor.a;

      if (x > 0 && !visited[idx - 1]) stack.push([x - 1, y]);
      if (x < width - 1 && !visited[idx + 1]) stack.push([x + 1, y]);
      if (y > 0 && !visited[idx - width]) stack.push([x, y - 1]);
      if (y < height - 1 && !visited[idx + width]) stack.push([x, y + 1]);
    }
  }

  maskCtx.putImageData(maskImageData, 0, 0);
}

/**
 * Region growing fill with Embedded Boundary Info.
 * Halts expanding when encountering a strong edge.
 */
export function regionGrowFillEdgeEmbedded(
  mainCtx: CanvasRenderingContext2D,
  maskCtx: CanvasRenderingContext2D,
  startX: number,
  startY: number,
  fillColor: { r: number; g: number; b: number; a: number },
  threshold: number,
  edgeThreshold: number
) {
  const width = mainCtx.canvas.width;
  const height = mainCtx.canvas.height;
  
  const mainImageData = mainCtx.getImageData(0, 0, width, height);
  const mainData = mainImageData.data;
  
  const maskImageData = maskCtx.getImageData(0, 0, width, height);
  const maskData = maskImageData.data;

  const edges = computeEdgeMap(mainData, width, height);
  const visited = new Uint8Array(width * height);

  const startPos = (startY * width + startX) * 4;
  const seedIntensity = (mainData[startPos] + mainData[startPos + 1] + mainData[startPos + 2]) / 3;

  const stack: [number, number][] = [[startX, startY]];

  while (stack.length > 0) {
    const [x, y] = stack.pop()!;
    const idx = y * width + x;
    
    if (visited[idx]) continue;
    visited[idx] = 1;

    const pos = idx * 4;
    const intensity = (mainData[pos] + mainData[pos + 1] + mainData[pos + 2]) / 3;
    const edgeMag = edges[idx];

    // Penalty: Stop if strong edge
    if (edgeMag > edgeThreshold) {
      continue;
    }

    if (Math.abs(intensity - seedIntensity) <= threshold) {
      maskData[pos] = fillColor.r;
      maskData[pos + 1] = fillColor.g;
      maskData[pos + 2] = fillColor.b;
      maskData[pos + 3] = fillColor.a;

      if (x > 0 && !visited[idx - 1]) stack.push([x - 1, y]);
      if (x < width - 1 && !visited[idx + 1]) stack.push([x + 1, y]);
      if (y > 0 && !visited[idx - width]) stack.push([x, y - 1]);
      if (y < height - 1 && !visited[idx + width]) stack.push([x, y + 1]);
    }
  }

  maskCtx.putImageData(maskImageData, 0, 0);
}

/**
 * Split-and-Merge with Contours
 * Approximated by a block-level segmentation and subsequent boundary-aware merge.
 */
export function regionGrowFillSplitMerge(
  mainCtx: CanvasRenderingContext2D,
  maskCtx: CanvasRenderingContext2D,
  startX: number,
  startY: number,
  fillColor: { r: number; g: number; b: number; a: number },
  threshold: number,
  edgeThreshold: number
) {
  const width = mainCtx.canvas.width;
  const height = mainCtx.canvas.height;
  
  const mainImageData = mainCtx.getImageData(0, 0, width, height);
  const mainData = mainImageData.data;
  const maskImageData = maskCtx.getImageData(0, 0, width, height);
  const maskData = maskImageData.data;

  const edges = computeEdgeMap(mainData, width, height);
  
  const MIN_SIZE = 4;
  const blockW = Math.ceil(width / MIN_SIZE);
  const blockH = Math.ceil(height / MIN_SIZE);
  const numBlocks = blockW * blockH;
  
  const blockMeans = new Float32Array(numBlocks);
  const blockEdgeMins = new Float32Array(numBlocks); 
  const parent = new Int32Array(numBlocks);
  
  for (let by = 0; by < blockH; by++) {
    for (let bx = 0; bx < blockW; bx++) {
      const bIdx = by * blockW + bx;
      parent[bIdx] = bIdx;
      
      let sum = 0;
      let count = 0;
      let maxEdge = 0;
      for (let y = by * MIN_SIZE; y < Math.min((by + 1) * MIN_SIZE, height); y++) {
        for (let x = bx * MIN_SIZE; x < Math.min((bx + 1) * MIN_SIZE, width); x++) {
          const idx = y * width + x;
          const pos = idx * 4;
          const intensity = (mainData[pos] + mainData[pos + 1] + mainData[pos + 2]) / 3;
          sum += intensity;
          if (edges[idx] > maxEdge) maxEdge = edges[idx];
          count++;
        }
      }
      blockMeans[bIdx] = sum / count;
      blockEdgeMins[bIdx] = maxEdge;
    }
  }
  
  function find(i: number): number {
    if (parent[i] !== i) {
      parent[i] = find(parent[i]);
    }
    return parent[i];
  }
  
  function union(i: number, j: number) {
    const rootI = find(i);
    const rootJ = find(j);
    if (rootI !== rootJ) {
      parent[rootI] = rootJ;
    }
  }
  
  for (let by = 0; by < blockH; by++) {
    for (let bx = 0; bx < blockW; bx++) {
      const u = by * blockW + bx;
      
      if (bx < blockW - 1) {
        const v = by * blockW + (bx + 1);
        if (blockEdgeMins[u] <= edgeThreshold && blockEdgeMins[v] <= edgeThreshold) {
          if (Math.abs(blockMeans[u] - blockMeans[v]) <= threshold) {
            union(u, v);
          }
        }
      }
      if (by < blockH - 1) {
        const v = (by + 1) * blockW + bx;
        if (blockEdgeMins[u] <= edgeThreshold && blockEdgeMins[v] <= edgeThreshold) {
          if (Math.abs(blockMeans[u] - blockMeans[v]) <= threshold) {
            union(u, v);
          }
        }
      }
    }
  }
  
  const startBx = Math.floor(startX / MIN_SIZE);
  const startBy = Math.floor(startY / MIN_SIZE);
  const seedBlockRoot = find(startBy * blockW + startBx);
  
  for (let by = 0; by < blockH; by++) {
    for (let bx = 0; bx < blockW; bx++) {
      if (find(by * blockW + bx) === seedBlockRoot) {
        for (let y = by * MIN_SIZE; y < Math.min((by + 1) * MIN_SIZE, height); y++) {
          for (let x = bx * MIN_SIZE; x < Math.min((bx + 1) * MIN_SIZE, width); x++) {
            const pos = (y * width + x) * 4;
            maskData[pos] = fillColor.r;
            maskData[pos + 1] = fillColor.g;
            maskData[pos + 2] = fillColor.b;
            maskData[pos + 3] = fillColor.a;
          }
        }
      }
    }
  }

  maskCtx.putImageData(maskImageData, 0, 0);
}

/**
 * Convert hex to RGBA
 */
export function hexToRgb(hex: string, alpha: number = 255) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16),
    a: alpha
  } : { r: 0, g: 0, b: 0, a: alpha };
}
