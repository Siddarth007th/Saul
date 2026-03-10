CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS legal_document_chunks (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(768),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX ON legal_document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
