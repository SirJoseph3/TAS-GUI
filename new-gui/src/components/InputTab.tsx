import React, { useCallback, useState } from 'react';
import { useSettingsStore } from '../stores/settingsStore';
import { toast } from 'sonner';
import { Film, Folder, Trash2 } from 'lucide-react';

const path = window.electronAPI?.path;

export const InputTab: React.FC = () => {
  const {
    inputPaths,
    outputPath,
    recursiveFolder,
    preserveFolderStructure,
    setInputPaths,
    setOutputPath,
    setRecursiveFolder,
    setPreserveFolderStructure,
    addInputPath,
    removeInputPath,
  } = useSettingsStore();
  
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleSelectInput = async () => {
    if (window.electronAPI?.dialog) {
      const result = await window.electronAPI.dialog.selectFiles({
        properties: ['openFile', 'openDirectory', 'multiSelections'],
        filters: [
          { name: 'Video Files', extensions: ['mp4', 'mkv', 'avi', 'mov', 'wmv', 'flv', 'webm', 'm4v'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });
      
      if (result) {
        result.forEach((path: string) => addInputPath(path));
      }
    } else {
      fileInputRef.current?.click();
    }
  };
  
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    files.forEach(file => {
      const path = (file as any).path || URL.createObjectURL(file);
      addInputPath(path);
    });
  };
  
  const handleSelectOutput = async () => {
    const result = await window.electronAPI?.dialog.selectFolder();
    if (result) {
      setOutputPath(result);
    }
  };
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    files.forEach((file) => {
      const filePath = (file as any).path;
      if (filePath) {
        addInputPath(filePath);
      }
    });
  }, [addInputPath]);
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  
  const getFileName = (fullPath: string) => {
    if (path) {
      return path.basename(fullPath);
    }
    return fullPath.split('\\').pop() || fullPath.split('/').pop() || fullPath;
  };
  
  const clearAll = () => {
    setInputPaths([]);
    setOutputPath('');
    toast.success('All inputs cleared');
  };
  
  return (
    <div className="space-y-6">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="video/*"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />
      
      {/* Input Section */}
      <div>
        <label className="block text-sm font-medium text-gray-400 mb-3">
          Input Files/Folders
        </label>
        
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`
            border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200
            ${isDragging
              ? 'border-[#008f91] bg-[#008f91]/10'
              : 'border-gray-600 hover:border-[#008f91]/50 bg-[#2b2b2b]'
            }
          `}
        >
          {inputPaths.length === 0 ? (
            <div className="space-y-4">
              <div className="w-16 h-16 mx-auto rounded-full bg-[#008f91]/20 flex items-center justify-center">
                <Film className="w-8 h-8 text-[#008f91]" />
              </div>
              <div>
                <p className="text-lg font-medium">Drop files or folders here</p>
                <p className="text-sm text-gray-500 mt-1">or click to browse</p>
              </div>
              <button
                onClick={handleSelectInput}
                className="px-6 py-2 bg-[#008f91] text-white rounded-lg hover:bg-[#008f91]/90 transition-colors"
              >
                Browse
              </button>
            </div>
          ) : (
            <div className="text-left space-y-3">
              <div className="flex justify-between items-center mb-4">
                <span className="text-sm text-gray-400">{inputPaths.length} item(s) selected</span>
                <button
                  onClick={clearAll}
                  className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>
              
              <div className="max-h-48 overflow-y-auto space-y-2 pr-2">
                {inputPaths.map((path, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-[#1a1a1a] rounded-lg group"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <Film className="w-5 h-5 text-[#008f91] flex-shrink-0" />
                      <span className="text-sm truncate" title={path}>
                        {getFileName(path)}
                      </span>
                    </div>
                    <button
                      onClick={() => removeInputPath(index)}
                      className="p-1.5 text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
              
              <button
                onClick={handleSelectInput}
                className="w-full py-2 border border-dashed border-gray-600 rounded-lg text-sm text-gray-400 hover:text-white hover:border-[#008f91] transition-colors"
              >
                + Add more files/folders
              </button>
            </div>
          )}
        </div>
        
        {/* Folder Options */}
        <div className="mt-4 space-y-3">
          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={recursiveFolder}
              onChange={(e) => setRecursiveFolder(e.target.checked)}
              className="w-5 h-5 rounded border-gray-600 bg-[#2b2b2b] text-[#008f91] focus:ring-[#008f91] focus:ring-offset-0"
            />
            <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
              Include subfolders
            </span>
          </label>
          
          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={preserveFolderStructure}
              onChange={(e) => setPreserveFolderStructure(e.target.checked)}
              className="w-5 h-5 rounded border-gray-600 bg-[#2b2b2b] text-[#008f91] focus:ring-[#008f91] focus:ring-offset-0"
            />
            <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
              Preserve folder structure
            </span>
          </label>
        </div>
      </div>
      
      {/* Output Section */}
      <div>
        <label className="block text-sm font-medium text-gray-400 mb-3">
          Output Destination
        </label>
        
        <div className="flex gap-3">
          <div className="flex-1 p-3 bg-[#2b2b2b] rounded-lg border border-gray-700 text-sm truncate">
            {outputPath || 'Save in the same folder as video (default)'}
          </div>
          <button
            onClick={handleSelectOutput}
            className="px-4 py-2 bg-[#2b2b2b] border border-gray-600 rounded-lg hover:border-[#008f91] transition-colors flex items-center gap-2"
          >
            <Folder className="w-4 h-4" />
            Browse
          </button>
        </div>
      </div>
    </div>
  );
};
