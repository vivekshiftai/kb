// API Types for PDF Processing
export interface PDFInfo {
  filename: string;
  upload_date?: string;
  page_count?: number;
  file_size?: number;
  chunk_count?: number;
}

export interface PDFListResponse {
  pdfs: PDFInfo[];
  total_count: number;
}

export interface QueryRequest {
  pdf_filename: string;
  query: string;
  max_results?: number;
}

export interface ImageInfo {
  filename: string;
  url: string;
  page_number: number;
}

export interface QueryResult {
  heading: string;
  text: string;
  score: number;
  page_number?: number;
  images: ImageInfo[];
}

export interface QueryResponse {
  pdf_filename: string;
  query: string;
  answer: string;
  results: QueryResult[];
  total_matches: number;
  processing_time: number;
}

export interface UploadResponse {
  success: boolean;
  message: string;
  pdf_filename: string;
  document_id?: string;
  processing_status: string;
}

export interface ErrorResponse {
  success: boolean;
  error: string;
  details?: Record<string, any>;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  message: string;
  vector_store_status: string;
  openai_status: string;
  timestamp: string;
}

// Processing States
export type ProcessingStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error';

export interface ProcessingState {
  status: ProcessingStatus;
  progress: number;
  message: string;
  error?: string;
}