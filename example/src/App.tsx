import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Brush, 
  Eraser, 
  PaintBucket, 
  Download, 
  Upload, 
  Trash2, 
  Undo2,
  ZoomIn,
  ZoomOut,
  Maximize,
  CheckCircle2,
  Settings2,
  Bone
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { Tool, Label, LABELS, FillAlgorithm } from './types';
import { regionGrowFillStandard, regionGrowFillEdgeEmbedded, regionGrowFillSplitMerge, hexToRgb } from './utils/canvasUtils';

export default function App() {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [activeTool, setActiveTool] = useState<Tool>('brush');
  const [activeLabel, setActiveLabel] = useState<Label>('femur');
  const [brushSize, setBrushSize] = useState(20);
  const [opacity, setOpacity] = useState(0.5);
  const [fillThreshold, setFillThreshold] = useState(5);
  const [fillAlgorithm, setFillAlgorithm] = useState<FillAlgorithm>('standard');
  const [edgeThreshold, setEdgeThreshold] = useState(30);
  const [isDrawing, setIsDrawing] = useState(false);
  const [baseScale, setBaseScale] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [history, setHistory] = useState<ImageData[]>([]);

  const mainCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const saveHistoryState = useCallback(() => {
    const canvas = maskCanvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setHistory(prev => [...prev.slice(-19), imageData]); // Keep last 20 states
  }, []);

  const handleUndo = () => {
    if (history.length > 1) {
      const newHistory = history.slice(0, -1);
      setHistory(newHistory);
      const canvas = maskCanvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (canvas && ctx) {
        ctx.putImageData(newHistory[newHistory.length - 1], 0, 0);
      }
    }
  };

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 4));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.25));
  const handleZoomReset = () => setZoom(1);

  // Initialize canvases
  useEffect(() => {
    if (!image || !mainCanvasRef.current || !maskCanvasRef.current) return;

    const mainCanvas = mainCanvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const mainCtx = mainCanvas.getContext('2d');
    const maskCtx = maskCanvas.getContext('2d');

    if (!mainCtx || !maskCtx) return;

    // Calculate scaling to fit viewport
    const container = containerRef.current;
    if (!container) return;

    const availableWidth = container.clientWidth - 40;
    const availableHeight = container.clientHeight - 40;
    
    const scale = Math.min(availableWidth / image.width, availableHeight / image.height);
    setBaseScale(scale);
    setZoom(1);
    
    mainCanvas.width = image.width;
    mainCanvas.height = image.height;
    maskCanvas.width = image.width;
    maskCanvas.height = image.height;

    mainCtx.drawImage(image, 0, 0);
    
    maskCtx.clearRect(0, 0, image.width, image.height);
    const initialImageData = maskCtx.getImageData(0, 0, image.width, image.height);
    setHistory([initialImageData]);
  }, [image]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => setImage(img);
      img.src = event.target?.result as string;
    };
    reader.readAsDataURL(file);
  };

  const getCanvasCoordinates = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    if (!image) return;
    const { x, y } = getCanvasCoordinates(e);
    
    const ctx = maskCanvasRef.current?.getContext('2d');
    const mainCtx = mainCanvasRef.current?.getContext('2d');
    if (!ctx || !mainCtx) return;

    if (activeTool === 'fill') {
      const labelConfig = LABELS.find(l => l.id === activeLabel);
      if (labelConfig) {
        const rgb = activeLabel === 'background' ? { r: 0, g: 0, b: 0, a: 0 } : hexToRgb(labelConfig.color);
        if (fillAlgorithm === 'standard') {
          regionGrowFillStandard(mainCtx, ctx, Math.floor(x), Math.floor(y), rgb, fillThreshold);
        } else if (fillAlgorithm === 'edge-embedded') {
          regionGrowFillEdgeEmbedded(mainCtx, ctx, Math.floor(x), Math.floor(y), rgb, fillThreshold, edgeThreshold);
        } else if (fillAlgorithm === 'split-merge') {
          regionGrowFillSplitMerge(mainCtx, ctx, Math.floor(x), Math.floor(y), rgb, fillThreshold, edgeThreshold);
        }
        saveHistoryState();
      }
      return;
    }

    setIsDrawing(true);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = brushSize;
    ctx.strokeStyle = (activeTool === 'eraser' || activeLabel === 'background') ? '#000000' : LABELS.find(l => l.id === activeLabel)?.color || '#ff0000';
    
    if (activeTool === 'eraser' || activeLabel === 'background') {
      ctx.globalCompositeOperation = 'destination-out';
    } else {
      ctx.globalCompositeOperation = 'source-over';
    }
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing || !image) return;
    const { x, y } = getCanvasCoordinates(e);
    
    const ctx = maskCanvasRef.current?.getContext('2d');
    if (!ctx) return;

    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    if (isDrawing) {
      setIsDrawing(false);
      saveHistoryState();
    }
  };

  const clearAnnotation = () => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
    saveHistoryState();
  };

  const downloadMask = () => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = 'bone_segmentation_mask.png';
    link.href = canvas.toDataURL();
    link.click();
  };

  return (
    <div className="flex h-screen w-full bg-medical-dark overflow-hidden selection:bg-medical-accent/30">
      {/* Sidebar */}
      <aside className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col p-6 gap-8 z-10 shadow-2xl">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-medical-accent/10 rounded-xl">
            <Bone className="w-6 h-6 text-medical-accent" />
          </div>
          <div>
            <h1 className="font-semibold text-lg tracking-tight">OrthoSeg</h1>
            <p className="text-xs text-slate-500 font-medium uppercase tracking-widest">Medical Imaging</p>
          </div>
        </div>

        {/* Upload Section */}
        {!image && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-slate-700 rounded-2xl bg-slate-800/50 hover:border-medical-accent/50 transition-colors cursor-pointer group"
            onClick={() => document.getElementById('file-upload')?.click()}
          >
            <Upload className="w-10 h-10 text-slate-500 group-hover:text-medical-accent mb-4 transition-colors" />
            <p className="text-sm font-medium text-slate-400 text-center">Upload X-ray to start</p>
            <input id="file-upload" type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
          </motion.div>
        )}

        {image && (
          <div className="flex flex-col gap-6">
            {/* Annotation Label Selector */}
            <div className="space-y-4">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Select Anatomy</label>
              <div className="grid grid-cols-1 gap-2">
                {LABELS.map((label) => (
                  <button
                    key={label.id}
                    onClick={() => setActiveLabel(label.id)}
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all border ${
                      activeLabel === label.id 
                      ? 'bg-slate-800 border-slate-700 ring-1 ring-medical-accent/50 text-white' 
                      : 'border-transparent text-slate-400 hover:bg-slate-800/50'
                    }`}
                  >
                    <div className={`w-3 h-3 rounded-full ${label.previewColor} shadow-[0_0_8px_rgba(0,0,0,0.5)]`} />
                    <span className="text-sm font-medium">{label.name}</span>
                    {activeLabel === label.id && <CheckCircle2 className="w-4 h-4 ml-auto text-medical-accent" />}
                  </button>
                ))}
              </div>
            </div>

            {/* Tools Selector */}
            <div className="space-y-4">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Toolbox</label>
              <div className="flex gap-2">
                {[
                  { id: 'brush', icon: Brush, label: 'Brush' },
                  { id: 'fill', icon: PaintBucket, label: 'Fill' },
                  { id: 'eraser', icon: Eraser, label: 'Eraser' }
                ].map((tool) => (
                  <button
                    key={tool.id}
                    onClick={() => setActiveTool(tool.id as Tool)}
                    className={`flex-1 flex flex-col items-center justify-center p-3 rounded-xl transition-all border ${
                      activeTool === tool.id 
                      ? 'bg-medical-accent text-white border-medical-accent/50 shadow-lg shadow-medical-accent/20' 
                      : 'bg-slate-800/30 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                    title={tool.label}
                  >
                    <tool.icon className="w-5 h-5" />
                  </button>
                ))}
              </div>
            </div>

            {/* Settings */}
            <div className="space-y-6 pt-4 border-t border-slate-800">
              {activeTool !== 'fill' && (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Brush Size</label>
                    <span className="text-xs font-mono text-medical-accent">{brushSize}px</span>
                  </div>
                  <input 
                    type="range" 
                    min="2" 
                    max="100" 
                    value={brushSize} 
                    onChange={(e) => setBrushSize(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-medical-accent"
                  />
                </div>
              )}

              {activeTool === 'fill' && (
                <div className="space-y-4">
                  <div className="space-y-3">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Algorithm</label>
                    <select
                      value={fillAlgorithm}
                      onChange={(e) => setFillAlgorithm(e.target.value as FillAlgorithm)}
                      className="w-full bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-2 outline-none focus:ring-1 focus:ring-medical-accent"
                    >
                      <option value="standard">Standard Growing</option>
                      <option value="edge-embedded">Embedded Boundary</option>
                      <option value="split-merge">Split-and-Merge</option>
                    </select>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Intensity Thresh</label>
                    <span className="text-xs font-mono text-medical-accent">{fillThreshold}</span>
                  </div>
                  <input 
                    type="range" 
                    min="1" 
                    max="50" 
                    value={fillThreshold} 
                    onChange={(e) => setFillThreshold(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-medical-accent mb-4"
                  />

                  {fillAlgorithm !== 'standard' && (
                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="pt-2">
                      <div className="flex justify-between items-center mb-4">
                        <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Edge Penalty</label>
                        <span className="text-xs font-mono text-medical-accent">{edgeThreshold}</span>
                      </div>
                      <input 
                        type="range" 
                        min="1" 
                        max="255" 
                        value={edgeThreshold} 
                        onChange={(e) => setEdgeThreshold(parseInt(e.target.value))}
                        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-medical-accent"
                      />
                    </motion.div>
                  )}
                </div>
              )}

              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Mask Opacity</label>
                  <span className="text-xs font-mono text-medical-accent">{Math.round(opacity * 100)}%</span>
                </div>
                <input 
                  type="range" 
                  min="0.1" 
                  max="1" 
                  step="0.1"
                  value={opacity} 
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-medical-accent"
                />
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col gap-2 pt-6">
              <button 
                onClick={downloadMask}
                className="flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-medical-accent hover:bg-medical-accent/90 text-slate-900 font-semibold transition-all"
              >
                <Download className="w-4 h-4" />
                Export Mask
              </button>
              <button 
                onClick={clearAnnotation}
                className="flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-slate-800 hover:bg-red-500/10 text-slate-400 hover:text-red-400 border border-slate-700 hover:border-red-500/50 transition-all"
              >
                <Trash2 className="w-4 h-4" />
                Clear All
              </button>
            </div>
          </div>
        )}
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col relative overflow-hidden" ref={containerRef}>
        <header className="h-16 px-8 border-b border-white/5 flex items-center justify-between bg-slate-900/50 backdrop-blur-md absolute top-0 left-0 right-0 z-20">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-[10px] uppercase tracking-wider font-bold text-slate-300">Active Session</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button 
              onClick={handleUndo} 
              disabled={history.length <= 1} 
              className="p-2 hover:bg-white/5 rounded-lg text-slate-400 disabled:opacity-30 disabled:cursor-not-allowed transition-all" 
              title="Undo Last Action"
            >
              <Undo2 className="w-5 h-5" />
            </button>
            <div className="w-px h-4 bg-slate-800 mx-2" />
            <button 
              onClick={handleZoomOut} 
              className="p-2 hover:bg-white/5 rounded-lg text-slate-400 transition-all" 
              title="Zoom Out"
            >
              <ZoomOut className="w-5 h-5" />
            </button>
            <button 
              onClick={handleZoomIn} 
              className="p-2 hover:bg-white/5 rounded-lg text-slate-400 transition-all" 
              title="Zoom In"
            >
              <ZoomIn className="w-5 h-5" />
            </button>
            <button 
              onClick={handleZoomReset} 
              className="p-2 hover:bg-white/5 rounded-lg text-slate-400 transition-all" 
              title="Reset Zoom"
            >
              <Maximize className="w-5 h-5" />
            </button>
          </div>
        </header>

        <div className="flex-1 flex items-center justify-center p-10 mt-16 overflow-auto">
          {image ? (
            <div className="relative shadow-[0_0_50px_rgba(0,0,0,0.5)] rounded-sm bg-black overflow-hidden ring-1 ring-white/10 group">
              <canvas 
                ref={mainCanvasRef}
                className="block"
                style={{ width: image.width * baseScale * zoom, height: image.height * baseScale * zoom }}
              />
              <canvas 
                ref={maskCanvasRef}
                className="absolute top-0 left-0 cursor-crosshair touch-none"
                style={{ opacity: opacity, width: image.width * baseScale * zoom, height: image.height * baseScale * zoom }}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />
              {/* Tool Tooltip */}
              <div className="absolute bottom-4 left-4 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
                <p className="text-[10px] font-mono bg-black/80 px-2 py-1 rounded text-medical-accent uppercase tracking-tighter">
                  {activeTool} tool | {activeLabel} | {brushSize}px
                </p>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4 max-w-sm text-center">
              <div className="w-20 h-20 rounded-full bg-slate-800/50 flex items-center justify-center mb-4">
                <Bone className="w-10 h-10 text-slate-600" />
              </div>
              <h2 className="text-xl font-semibold tracking-tight">Ready for segmentation</h2>
              <p className="text-slate-400 text-sm leading-relaxed">
                Drag and drop your DICOM export or standard X-ray image (PNG/JPG) to begin annotating the lower limb anatomy.
              </p>
              <button 
                onClick={() => document.getElementById('file-upload')?.click()}
                className="mt-4 px-6 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-full transition-all border border-slate-700"
              >
                Browse Files
              </button>
            </div>
          )}
        </div>

        {/* Footer Info */}
        <footer className="h-10 px-8 flex items-center justify-between text-[10px] font-mono text-slate-500 bg-slate-900/80 border-t border-white/5">
          <div className="flex gap-4">
            <span>READY_FOR_INPUT</span>
            <span>V_1.0.0</span>
          </div>
          <div className="flex gap-4">
            <span>{image ? `${image.width}x${image.height}` : 'NO_IMAGE'}</span>
            <span className="flex items-center gap-1">ZOOM <div className="w-1 h-1 rounded-full bg-medical-accent" /> {Math.round(zoom * 100)}%</span>
            <span className="flex items-center gap-1">MB <div className="w-1 h-1 rounded-full bg-slate-600" /> 0.0</span>
          </div>
        </footer>
      </main>
    </div>
  );
}
