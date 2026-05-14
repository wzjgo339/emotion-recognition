import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

export default function ImageUpload({ image, onImageSelect, disabled }) {
  const onDrop = useCallback(
    (accepted) => {
      if (accepted.length > 0) {
        onImageSelect(accepted[0]);
      }
    },
    [onImageSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
    disabled,
  });

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400 bg-white'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />

        {image ? (
          <div className="space-y-3">
            <img
              src={URL.createObjectURL(image)}
              alt="preview"
              className="max-h-64 mx-auto rounded-lg object-contain"
            />
            <p className="text-sm text-gray-500">{image.name}</p>
            <span className="inline-block text-xs text-blue-500">
              Click or drag to replace
            </span>
          </div>
        ) : (
          <div className="space-y-3 py-8">
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <p className="text-gray-500 text-sm">
              {isDragActive
                ? 'Drop image here'
                : 'Drag & drop an image here, or click to select'}
            </p>
            <p className="text-xs text-gray-400">JPG / PNG / BMP, max 10MB</p>
          </div>
        )}
      </div>
    </div>
  );
}
