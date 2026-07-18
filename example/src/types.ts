export type Tool = 'brush' | 'eraser' | 'fill';
export type Label = 'background' | 'femur' | 'tibia' | 'fibula';
export type FillAlgorithm = 'standard' | 'edge-embedded' | 'split-merge';

export interface LabelConfig {
  id: Label;
  name: string;
  color: string; // Hex color for drawing
  previewColor: string; // Tailwind color for UI
}

export const LABELS: LabelConfig[] = [
  { id: 'background', name: 'Background', color: '#000000', previewColor: 'bg-transparent border-2 border-slate-500 border-dashed' },
  { id: 'femur', name: 'Femur', color: '#ef4444', previewColor: 'bg-red-500' },
  { id: 'tibia', name: 'Tibia', color: '#22c55e', previewColor: 'bg-green-500' },
  { id: 'fibula', name: 'Fibula', color: '#3b82f6', previewColor: 'bg-blue-500' }
];
